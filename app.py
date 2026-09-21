import os
import hashlib
import json
import chromadb
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from sentence_transformers import SentenceTransformer
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
client = chromadb.PersistentClient(path="./chroma_webapp_db")
collection = client.get_or_create_collection(
    name="webapp_documents",
    metadata={"hnsw:space": "cosine"}
)

def index_file(filepath):
    """
    Indexes a file into Chroma. Returns the number of chunks indexed.
    Returns 0 if the file's content is an exact duplicate of content
    already indexed under a different filename (nothing new indexed).
    """
    raw_text = load_text(filepath)
    filename = os.path.basename(filepath)
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    # Skip indexing if this exact content already exists under a
    # *different* source filename (cheap exact-duplicate guard; does
    # not catch near-duplicates or paraphrased content — that's a
    # later, smarter-RAG improvement).
    existing = collection.get(where={"content_hash": content_hash})
    if existing["ids"] and not all(
        meta.get("source") == filename for meta in existing["metadatas"]
    ):
        return 0

    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    embeddings = embedder.encode(chunks).tolist()

    # Remove any existing chunks for this filename before re-indexing,
    # so a shorter re-upload doesn't leave orphaned old chunks behind.
    collection.delete(where={"source": filename})

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "content_hash": content_hash} for _ in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)

def retrieve(query, top_k=3, source_filter=None, max_distance=1.0):
    query_embedding = embedder.encode(query).tolist()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)
    matched_chunks = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]  # carries the "source" filename per chunk

    # Drop chunks that are too semantically distant to be useful context.
    # max_distance is a loose starting cutoff for cosine distance — tune
    # it once you've seen how it behaves on real queries.
    filtered = [
        (d, c, m.get("source", "unknown"))
        for d, c, m in zip(distances, matched_chunks, metadatas)
        if d <= max_distance
    ]
    return filtered

def jsonify_sources(retrieved_chunks):
    """
    Serializes retrieved chunks into a JSON string for storage in
    Message.sources_json. This is NOT Flask's jsonify (which builds an
    HTTP response) — just Python's json.dumps, packaging the source
    info so it can be stored as plain text in the database and parsed
    back out later for display (source citations feature).
    """
    return json.dumps([
        {"distance": round(distance, 4), "text": chunk, "source": source}
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
            "model": "openrouter/free",
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

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        num_chunks = index_file(filepath)
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
    # Chroma has no built-in "distinct values" query, so pull all metadata
    # and de-duplicate in Python. Fine at small/demo scale.
    all_items = collection.get()
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

    # Remove all chunks tagged with this source from the vector database.
    existing = collection.get(where={"source": filename})
    if not existing["ids"]:
        return jsonify({"error": f"No indexed document found named '{filename}'."}), 404

    collection.delete(where={"source": filename})

    # Also remove the physical file from disk, so it doesn't linger and
    # doesn't get accidentally re-served or re-indexed later. Missing on
    # disk (e.g. already removed manually) is not treated as a failure —
    # the important part (removing it from search) already succeeded.
    filepath = os.path.join(UPLOAD_FOLDER, filename)
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
        results = retrieve(query, top_k=3, source_filter=source_filter)
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
        {"distance": round(distance, 4), "text": chunk, "source": source}
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