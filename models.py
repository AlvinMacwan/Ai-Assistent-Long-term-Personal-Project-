from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import MetaData

# A naming convention ensures every constraint (foreign keys, unique
# constraints, etc.) gets an automatic, predictable name instead of an
# anonymous one. This matters because SQLite migrations that rebuild a
# table (which Alembic does for most schema changes on SQLite) need
# every constraint to have a name to reference during the rebuild —
# without this, certain migrations fail outright with "Constraint must
# have a name". This is considered standard practice for any real
# Flask-SQLAlchemy project using migrations, not just a one-off fix.
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=convention)
db = SQLAlchemy(metadata=metadata)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    messages = db.relationship("Message", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Conversation(db.Model):
    __tablename__ = "conversations"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(200), nullable=False, default="New conversation")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # cascade="all, delete-orphan" means: if a Conversation row is deleted,
    # SQLAlchemy automatically deletes all its Message rows too, instead of
    # leaving orphaned messages pointing at a conversation that no longer
    # exists. Without this, deleting a conversation would either fail (if
    # the database enforces the foreign key) or silently leave broken data.
    messages = db.relationship(
        "Message", backref="conversation", lazy=True,
        cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    # Now a real foreign key into conversations.id, instead of a
    # self-generated string label. This means the database itself
    # enforces that every message belongs to a conversation that
    # actually exists.
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # "user" or "assistant"
    content = db.Column(db.Text, nullable=False)
    # Only populated for assistant messages that cited sources; stored as
    # JSON text (filename + distance per chunk) rather than a separate table,
    # since it's just for display and doesn't need to be queried on its own.
    sources_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)