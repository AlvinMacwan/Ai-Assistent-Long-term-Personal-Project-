import os
import re
import hashlib
import json
import chromadb
import requests
from rank_bm25 import BM25Okapi
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from docx import Document as DocxDocument
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from flask_migrate import Migrate

from models import db, User, Message, Conversation

load_dotenv()  # loads variables from a local .env file into the environment

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

app = Flask(__name__)

# --- Database config (Step 1 of V2) ---
# SECRET_KEY is required by Flask for session signing (Flask-Login will
# need this once auth is wired in next). Set a real value in your .env —
# the fallback here is only for local dev convenience.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-this")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
db.init_app(app)

# --- Migrations (Step 1 of V3's proper schema handling) ---
# From now on, schema changes go through migration files instead of
# db.create_all(). create_all() only adds brand-new tables and never
# modifies existing ones — Migrate lets us evolve the schema (add
# columns, change types, add tables) without ever deleting real data.
migrate = Migrate(app, db)

# --- Login setup (Step 2 of V2) ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # redirect target when @login_required fails


@login_manager.user_loader
def load_user(user_id):
    # Flask-Login calls this on every request to reload the logged-in
    # user from the session. Must return None (not raise) if not found.
    return User.query.get(int(user_id))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}

# Cosine distance cutoff for retrieval relevance. Chunks with a distance
# above this are dropped rather than passed to the LLM as context.
# Value chosen from real observed data across our own eval runs: every
# genuinely relevant chunk we've seen sits below ~0.8, while every
# confirmed-irrelevant chunk sits above ~0.92 — this threshold sits in
# that gap. Previously set to 1.0 (effectively no filtering), which let
# irrelevant chunks reach the LLM and contributed to at least one
# observed hallucination on an out-of-scope question. Re-tune this by
# re-running eval_set.json / run_eval.py if retrieval quality on new
# documents suggests the gap has shifted.
RELEVANCE_THRESHOLD = 0.8

# How many candidates to pull from Chroma before reranking. This needs
# to be meaningfully larger than top_k so the cross-encoder actually has
# room to promote a chunk the bi-encoder ranked outside the final cut.
# If a user's document set has fewer total chunks than this, Chroma just
# returns everything it has — not an error.
RERANK_CANDIDATE_POOL = 20

# ==========================================
# RAG pipeline
# ==========================================
def load_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".txt", ".md"):
        # Markdown is just plain text with formatting symbols (#, *, etc.)
        # left in place — no special parsing needed for this to work
        # fine as retrievable content; the symbols just sit in the text.
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    elif ext == ".docx":
        # .docx files are internally a zip of XML files. python-docx
        # handles that structure for us; we just walk the paragraphs
        # and join their text, same as the PDF page loop above.
        doc = DocxDocument(filepath)
        text = "\n".join(p.text for p in doc.paragraphs if p.text)
        return text
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text, chunk_size=500, overlap=50):
    # Splits on word boundaries (not raw characters) so chunks never cut
    # a word in half. chunk_size/overlap are treated as approximate
    # character counts and converted to a word count using a rough
    # average of ~6 characters per word (including spaces).
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size // 6
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words).strip())
        overlap_words = max(1, overlap // 6)
        start = end - overlap_words
    return [c for c in chunks if c]

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Cross-encoder used to rerank the retrieved candidate pool. Unlike the
# bi-encoder above (which embeds query and chunk separately, then
# compares vectors — fast, used for the initial Chroma search), the
# cross-encoder takes the query and a chunk together as one input and
# outputs a single relevance score straight from that joint comparison.
# More accurate, but too slow to run against the whole collection —
# hence: bi-encoder for broad retrieval, cross-encoder for reranking a
# small shortlist.
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

client = chromadb.PersistentClient(path="./chroma_webapp_db")
collection = client.get_or_create_collection(
    name="webapp_documents",
    metadata={"hnsw:space": "cosine"}
)

# ==========================================
# BM25 keyword search (hybrid search, alongside vector search)
# ==========================================
# Vector search (via Chroma above) is good at semantic similarity but
# can miss exact-term matches — acronyms, proper nouns, specific
# numbers — since embeddings capture meaning, not exact tokens. BM25
# is the opposite: strong on exact keyword matches, no notion of
# meaning. Running both and merging their candidates (see retrieve()
# below) gives semantic recall AND exact-match precision.
#
# Unlike Chroma, rank_bm25's BM25Okapi isn't a live index you insert
# into — it's built once from a full corpus and has to be rebuilt
# whenever that corpus changes. So we keep a plain in-memory list
# mirroring what's in Chroma, and rebuild the BM25 index whenever a
# document is added or removed.
bm25_corpus = []   # list of {"text": ..., "user_id": ..., "source": ...}
bm25_index = None  # rebuilt by _rebuild_bm25_index() whenever bm25_corpus changes


def _tokenize(text):
    # Simple lowercase word-splitting — enough for BM25's term-matching
    # purposes here; no stemming/lemmatization needed at this scale.
    return re.findall(r"\w+", text.lower())


def _rebuild_bm25_index():
    global bm25_index
    if bm25_corpus:
        bm25_index = BM25Okapi([_tokenize(item["text"]) for item in bm25_corpus])
    else:
        bm25_index = None


def _load_bm25_corpus_from_chroma():
    # Chroma is persistent on disk; this in-memory bm25_corpus list is
    # not, so on every app startup we rebuild it from whatever's
    # already indexed in Chroma to keep the two in sync.
    global bm25_corpus
    all_items = collection.get()
    bm25_corpus = [
        {"text": doc, "user_id": meta.get("user_id"), "source": meta.get("source")}
        for doc, meta in zip(all_items["documents"], all_items["metadatas"])
    ]
    _rebuild_bm25_index()


def _bm25_remove_entries(user_id, filename):
    # Mirrors collection.delete(where={"source": filename, "user_id": user_id})
    # — call this alongside that, so the two indexes never drift apart.
    global bm25_corpus
    bm25_corpus = [
        item for item in bm25_corpus
        if not (item["user_id"] == user_id and item["source"] == filename)
    ]


def _bm25_add_entries(chunks, user_id, filename):
    # Mirrors collection.upsert(...) — call this alongside that.
    for chunk in chunks:
        bm25_corpus.append({"text": chunk, "user_id": user_id, "source": filename})


def bm25_search(query, user_id, source_filter=None, top_n=None):
    """
    Keyword search scoped to a user's own documents, same isolation
    guarantee as retrieve()'s vector search. Returns a list of
    (chunk_text, source) tuples, ranked by BM25 score (higher = more
    relevant) — no distance metric, since BM25 scores aren't
    comparable to cosine distance.
    """
    if bm25_index is None or not bm25_corpus:
        return []

    if top_n is None:
        top_n = RERANK_CANDIDATE_POOL

    scores = bm25_index.get_scores(_tokenize(query))
    scored = [
        (score, item) for score, item in zip(scores, bm25_corpus)
        if item["user_id"] == user_id and (source_filter is None or item["source"] == source_filter)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [(item["text"], item["source"]) for _, item in scored[:top_n]]


_load_bm25_corpus_from_chroma()


def index_file(filepath, user_id):
    """
    Indexes a file into Chroma, scoped to a specific user. Returns the
    number of chunks indexed. Returns 0 if this exact content already
    exists under a different filename for the SAME user (duplicate
    check is per-user — one user's content hash should never be
    compared against another user's, since that would both leak
    whether another user has a given file and risk false-positive
    skips across accounts that should be fully isolated).
    """
    raw_text = load_text(filepath)
    filename = os.path.basename(filepath)
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    # Duplicate check scoped to this user only — $and combines both
    # conditions, since Chroma's `where` only ANDs top-level keys
    # implicitly in some versions; being explicit avoids ambiguity.
    existing = collection.get(where={
        "$and": [{"content_hash": content_hash}, {"user_id": user_id}]
    })
    if existing["ids"] and not all(
        meta.get("source") == filename for meta in existing["metadatas"]
    ):
        return 0

    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    embeddings = embedder.encode(chunks).tolist()

    # Remove any existing chunks for THIS filename AND THIS user before
    # re-indexing. Without the user_id here, one user re-uploading a
    # file could wipe out another user's identically-named file.
    collection.delete(where={
        "$and": [{"source": filename}, {"user_id": user_id}]
    })
    # Mirror the same removal in the BM25 corpus, so a re-upload doesn't
    # leave stale duplicate entries sitting alongside the fresh ones.
    _bm25_remove_entries(user_id, filename)

    # user_id is baked into the chunk ID itself, not just the metadata.
    # This is what actually prevents two different users' same-named
    # files from colliding on the same Chroma ID and overwriting each
    # other via upsert.
    ids = [f"user{user_id}_{filename}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": filename, "content_hash": content_hash, "user_id": user_id}
        for _ in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    # Mirror the same chunks into the BM25 corpus and rebuild its index
    # — BM25Okapi has no incremental "add one doc" operation, so any
    # corpus change means a full rebuild. Fine at this document scale.
    _bm25_add_entries(chunks, user_id, filename)
    _rebuild_bm25_index()

    return len(chunks)

def retrieve(query, user_id, top_k=3, source_filter=None, max_distance=RELEVANCE_THRESHOLD,
             candidate_pool=RERANK_CANDIDATE_POOL):
    query_embedding = embedder.encode(query).tolist()

    # user_id filtering is never optional — every retrieval must be
    # scoped to the requesting user's own documents. source_filter is
    # an additional, optional narrowing on top of that (a specific
    # file within this user's own documents).
    where_conditions = [{"user_id": user_id}]
    if source_filter:
        where_conditions.append({"source": source_filter})

    query_kwargs = {
        "query_embeddings": [query_embedding],
        # Over-fetch relative to top_k: pull a wider candidate pool than
        # we'll actually return, so the cross-encoder below has real
        # alternatives to promote instead of just re-sorting the same
        # top_k the bi-encoder already picked.
        "n_results": candidate_pool,
        "where": {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0],
    }

    results = collection.query(**query_kwargs)
    matched_chunks = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]  # carries the "source" filename per chunk

    # Drop chunks that are too semantically distant to be useful context.
    # See RELEVANCE_THRESHOLD's definition above for the reasoning
    # behind the default value. This filter runs on the wider candidate
    # pool, same as it used to run on the final top_k — just more chunks
    # to check now.
    vector_candidates = [
        (d, c, m.get("source", "unknown"))
        for d, c, m in zip(distances, matched_chunks, metadatas)
        if d <= max_distance
    ]

    # --- Hybrid search: bring in BM25 keyword matches (NEW) ---
    # Vector search can miss chunks that share no semantic "shape" with
    # the query but do contain its exact terms (acronyms, proper nouns,
    # specific numbers) — BM25 catches those. We don't threshold-filter
    # BM25 hits the way we do vector hits, since BM25 scores aren't on
    # the same 0-2 cosine-distance scale; instead we let the reranker
    # below judge relevance for every candidate, vector- or
    # keyword-sourced, using one consistent signal.
    bm25_matches = bm25_search(query, user_id, source_filter=source_filter, top_n=candidate_pool)

    # Union + dedupe by chunk text. Vector candidates already carry a
    # real cosine distance; BM25-only matches (found by keyword search
    # but absent from the vector search's own candidate pool) get
    # distance=None, since there's no cosine distance to report for a
    # chunk the vector search never actually surfaced.
    seen_chunks = {c for _, c, _ in vector_candidates}
    combined = list(vector_candidates)
    for chunk_text, source in bm25_matches:
        if chunk_text not in seen_chunks:
            combined.append((None, chunk_text, source))
            seen_chunks.add(chunk_text)

    if not combined:
        return []

    # --- Reranking ---
    # Score each surviving (query, chunk) pair with the cross-encoder.
    # Higher score = more relevant. This re-sorts the candidate pool
    # using a query-aware signal the bi-encoder's cosine distance can't
    # provide, then we cut down to top_k only after reranking.
    pairs = [(query, chunk) for _, chunk, _ in combined]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(zip(combined, rerank_scores), key=lambda pair: pair[1], reverse=True)

    # Drop the rerank score before returning — keeps the return shape
    # identical to before ((distance, chunk, source) tuples, distance
    # possibly None for BM25-only matches), so generate_answer() and
    # the /ask route keep working with only the small None-handling
    # tweak made to jsonify_sources() and the chunks_payload below.
    top_results = [item for item, score in reranked[:top_k]]
    return top_results

def jsonify_sources(retrieved_chunks):
    """
    Serializes retrieved chunks into a JSON string for storage in
    Message.sources_json. This is NOT Flask's jsonify (which builds an
    HTTP response) — just Python's json.dumps, packaging the source
    info so it can be stored as plain text in the database and parsed
    back out later for display (source citations feature).
    """
    return json.dumps([
        {
            # distance is None for chunks found only via BM25 keyword
            # search (no cosine distance exists for those), so guard
            # the round() call rather than let it crash on None.
            "distance": round(distance, 4) if distance is not None else None,
            "text": chunk,
            "source": source,
        }
        for distance, chunk, source in retrieved_chunks
    ])


def get_recent_history(conversation_id, limit=6):
    """
    Fetches the most recent messages for a conversation, oldest-first,
    so they can be replayed into the prompt in the order they happened.
    limit=6 means "last 3 question/answer pairs" (each pair is 2 rows:
    one 'user' role, one 'assistant' role).
    """
    recent = (
        Message.query
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(recent))  # flip back to chronological order


def generate_answer(query, retrieved_chunks, history=None):
    context = "\n\n".join([chunk for distance, chunk, source in retrieved_chunks])

    # Turn stored Message rows into plain "Role: text" lines the LLM can
    # read as prior conversation turns. If there's no history yet (first
    # message), this section is simply left empty.
    history_text = ""
    if history:
        lines = []
        for msg in history:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        history_text = "\n".join(lines)

    prompt = f"""Use the following context to answer the question. If the answer isn't in the context, say you don't know.

Context:
{context}

Previous conversation:
{history_text if history_text else "(none yet)"}

Question: {query}"""

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            # Pinned to a specific model rather than the "openrouter/free"
            # wildcard router. openrouter/free selects a DIFFERENT
            # underlying free model at random per request, which is why
            # eval runs showed inconsistent behavior (occasional garbled
            # non-answers, inconsistent refusal behavior) even with zero
            # code changes between runs. Pinning trades "always whatever's
            # available" for consistent, known behavior.
            # Switched from nvidia/nemotron-3-ultra-550b-a55b:free to
            # Qwen's free-tier model. Verified live on openrouter.ai
            # (compare page returns real metrics, not "no endpoints
            # found") — free, 262K context.
            # NOTE: free model availability changes often, and this
            # project has already hit the openrouter daily free-tier
            # rate limit once — if this ID ever errors as unavailable
            # or rate-limited, re-check openrouter.ai/models directly.
            "model": "qwen/qwen3.8-27b:free",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    data = response.json()

    if "choices" not in data:
        # Surface the API's own error message instead of crashing with a KeyError
        error_msg = data.get("error", {}).get("message", "Unknown error from OpenRouter")
        raise RuntimeError(error_msg)

    return data["choices"][0]["message"]["content"]


# ==========================================
# Routes
# ==========================================
@app.route("/")
@login_required
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("That username is already taken.")
            return redirect(url_for("register"))

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user is None or not user.check_password(password):
            flash("Invalid username or password.")
            return redirect(url_for("login"))

        login_user(user)
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    # Each user's uploads live in their own subfolder, keyed by user ID.
    # Without this, two users uploading a same-named file would silently
    # overwrite each other's physical file on disk, even after Chroma
    # itself is correctly isolated by user_id.
    user_upload_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))
    os.makedirs(user_upload_folder, exist_ok=True)
    filepath = os.path.join(user_upload_folder, filename)
    file.save(filepath)

    try:
        num_chunks = index_file(filepath, current_user.id)
    except Exception as e:
        return jsonify({"error": f"Failed to index file: {str(e)}"}), 500

    if num_chunks == 0:
        return jsonify({
            "message": f"'{filename}' matches content already indexed under another file — skipped."
        })

    return jsonify({
        "message": f"Indexed '{filename}' into {num_chunks} chunks."
    })

@app.route("/documents", methods=["GET"])
@login_required
def list_documents():
    # Only pull metadata belonging to the current user — otherwise this
    # would list every user's filenames mixed together.
    all_items = collection.get(where={"user_id": current_user.id})
    sources = set()
    for meta in all_items["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return jsonify({"documents": sorted(sources)})

@app.route("/documents/<filename>", methods=["DELETE"])
@login_required
def delete_document(filename):
    # secure_filename here guards the same path-traversal risk as on
    # upload — filename comes from the URL, which is user-controlled.
    filename = secure_filename(filename)

    where_clause = {"$and": [{"source": filename}, {"user_id": current_user.id}]}

    # Remove all chunks tagged with this source AND belonging to this
    # user — without the user_id condition, this would also match (and
    # delete) another user's identically-named file.
    existing = collection.get(where=where_clause)
    if not existing["ids"]:
        return jsonify({"error": f"No indexed document found named '{filename}'."}), 404

    collection.delete(where=where_clause)

    # Mirror the same removal in the BM25 corpus — otherwise a deleted
    # document would still surface via keyword search even though
    # vector search (and the UI's document list) no longer sees it.
    _bm25_remove_entries(current_user.id, filename)
    _rebuild_bm25_index()

    # Physical files now also live in a per-user subfolder (see /upload),
    # so this can never touch another user's file of the same name.
    filepath = os.path.join(UPLOAD_FOLDER, str(current_user.id), filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    return jsonify({"message": f"Deleted '{filename}' and its indexed chunks."})

@app.route("/conversations", methods=["GET"])
@login_required
def list_conversations():
    # Ownership is enforced right in the query itself (filter_by user_id),
    # not checked afterward — a user can only ever see their own rows
    # come back from this query in the first place.
    conversations = (
        Conversation.query
        .filter_by(user_id=current_user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    return jsonify({
        "conversations": [
            {"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()}
            for c in conversations
        ]
    })


@app.route("/conversations", methods=["POST"])
@login_required
def create_conversation():
    conversation = Conversation(user_id=current_user.id, title="New conversation")
    db.session.add(conversation)
    db.session.commit()
    return jsonify({"id": conversation.id, "title": conversation.title})


@app.route("/conversations/<int:conversation_id>/messages", methods=["GET"])
@login_required
def get_conversation_messages(conversation_id):
    # Explicit ownership check: fetch the conversation filtered by BOTH
    # its id AND the current user's id. If it exists but belongs to
    # someone else, this returns None just like it doesn't exist at all
    # — the requester can't tell the difference, which is the correct,
    # safe behavior (never reveal "that exists, but isn't yours").
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=current_user.id
    ).first()
    if conversation is None:
        return jsonify({"error": "Conversation not found."}), 404

    return jsonify({
        "id": conversation.id,
        "title": conversation.title,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "sources": json.loads(m.sources_json) if m.sources_json else None,
            }
            for m in conversation.messages  # already ordered by created_at via the model's relationship
        ],
    })


@app.route("/conversations/<int:conversation_id>", methods=["DELETE"])
@login_required
def delete_conversation(conversation_id):
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=current_user.id
    ).first()
    if conversation is None:
        return jsonify({"error": "Conversation not found."}), 404

    # The cascade="all, delete-orphan" set on Conversation.messages in
    # models.py means deleting this row also deletes all its Message
    # rows automatically — no need to manually delete them first.
    db.session.delete(conversation)
    db.session.commit()
    return jsonify({"message": "Conversation deleted."})


@app.route("/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json()
    query = data.get("question", "")
    source_filter = data.get("source") or None  # empty string -> None (no filter)
    conversation_id = data.get("conversation_id")

    if not query:
        return jsonify({"error": "No question provided"}), 400
    if not conversation_id:
        return jsonify({"error": "No conversation_id provided."}), 400

    # Same ownership check as the other conversation routes — never trust
    # a conversation_id from the request body without verifying it's
    # actually this user's conversation.
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=current_user.id
    ).first()
    if conversation is None:
        return jsonify({"error": "Conversation not found."}), 404

    try:
        history = get_recent_history(conversation.id)
        results = retrieve(query, current_user.id, top_k=3, source_filter=source_filter)
        if not results:
            return jsonify({"error": "No indexed documents match that filter, or no results were relevant enough."}), 400

        answer = generate_answer(query, results, history=history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Auto-title the conversation from its first question, so the
    # sidebar shows something meaningful instead of "New conversation"
    # for every entry. Only do this once — check via the default title,
    # so later messages in the same conversation don't keep overwriting it.
    if conversation.title == "New conversation":
        conversation.title = (query[:40] + "...") if len(query) > 40 else query

    # Persist this turn (both sides) so it's available as history on the
    # next question in this conversation.
    user_msg = Message(
        user_id=current_user.id,
        conversation_id=conversation.id,
        role="user",
        content=query,
    )
    assistant_msg = Message(
        user_id=current_user.id,
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources_json=jsonify_sources(results),
    )
    db.session.add(user_msg)
    db.session.add(assistant_msg)
    db.session.commit()

    # Return retrieved chunks alongside the answer, for transparency in the UI
    chunks_payload = [
        {
            # Same None-guard as jsonify_sources() — BM25-only matches
            # have no cosine distance to report.
            "distance": round(distance, 4) if distance is not None else None,
            "text": chunk,
            "source": source,
        }
        for distance, chunk, source in results
    ]

    # A clean, deduplicated list of just the source filenames used —
    # this is the actual "citation" list (e.g. "Sources: doc1.pdf, doc2.txt"),
    # separate from chunks_payload which keeps the full text for anyone
    # who wants to expand and see the exact retrieved passage.
    cited_sources = sorted(set(source for _, _, source in results))

    return jsonify({
        "answer": answer,
        "retrieved_chunks": chunks_payload,
        "sources": cited_sources,
    })


if __name__ == "__main__":
    app.run(debug=True) 