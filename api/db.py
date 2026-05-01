"""MySQL persistence for users, auth sessions, and mediation history."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import os
import re
import secrets
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Mapping

# Import config first so the project-local .env is loaded before MySQL settings are read.
from config import PROJECT_ROOT as _PROJECT_ROOT  # noqa: F401


PASSWORD_ITERATIONS = 310_000
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "concordai")


class DuplicateUserError(ValueError):
    """Raised when a user signs up with an email that already exists."""


def _mysql_connector():
    try:
        return importlib.import_module("mysql.connector")
    except ImportError as exc:  # pragma: no cover - environment setup guard
        raise RuntimeError(
            "mysql-connector-python is required. Install backend requirements before starting the API."
        ) from exc


def _safe_database_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise RuntimeError("MYSQL_DATABASE may only contain letters, numbers, and underscores.")
    return name


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


def _iso(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _connect(*, database: bool = True):
    connector = _mysql_connector()
    kwargs: dict[str, Any] = {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "autocommit": False,
    }
    if database:
        kwargs["database"] = MYSQL_DATABASE
    return connector.connect(**kwargs)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str, salt_hex: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        PASSWORD_ITERATIONS,
    )
    return digest.hex()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _user_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    created_at = row["created_at"]
    if hasattr(created_at, "isoformat"):
        created_at = created_at.isoformat()
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "created_at": created_at,
    }


def _ensure_resolution_feedback_schema(cursor: Any) -> None:
    cursor.execute(
        """
        SELECT COLUMN_NAME, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'conversation_resolutions'
        """,
        (MYSQL_DATABASE,),
    )
    columns = {name: nullable for name, nullable in cursor.fetchall()}
    if columns.get("user_a_rating") == "NO":
        cursor.execute("ALTER TABLE conversation_resolutions MODIFY user_a_rating TINYINT NULL")
    if columns.get("user_b_rating") == "NO":
        cursor.execute("ALTER TABLE conversation_resolutions MODIFY user_b_rating TINYINT NULL")
    if "user_a_comment" not in columns:
        cursor.execute("ALTER TABLE conversation_resolutions ADD COLUMN user_a_comment TEXT NULL AFTER note")
    if "user_b_comment" not in columns:
        cursor.execute("ALTER TABLE conversation_resolutions ADD COLUMN user_b_comment TEXT NULL AFTER user_a_comment")


def init_db() -> None:
    """Create the demo database and app tables if they do not exist."""
    database_name = _safe_database_name(MYSQL_DATABASE)
    with closing(_connect(database=False)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()

    with closing(_connect()) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id CHAR(36) PRIMARY KEY,
                    name VARCHAR(160) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash CHAR(64) NOT NULL,
                    password_salt CHAR(32) NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token_hash CHAR(64) PRIMARY KEY,
                    user_id CHAR(36) NOT NULL,
                    created_at DATETIME NOT NULL,
                    last_seen_at DATETIME NOT NULL,
                    CONSTRAINT fk_auth_tokens_user
                        FOREIGN KEY (user_id) REFERENCES users(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS mediation_history (
                    id CHAR(36) PRIMARY KEY,
                    user_id CHAR(36) NOT NULL,
                    conversation_id VARCHAR(80) NOT NULL,
                    turn INT NOT NULL,
                    mode VARCHAR(40) NOT NULL,
                    request_id CHAR(64) NOT NULL,
                    trace_id VARCHAR(80) NOT NULL,
                    text_a TEXT NOT NULL,
                    text_b TEXT NOT NULL,
                    response_a TEXT NOT NULL,
                    response_b TEXT NOT NULL,
                    conversation_status VARCHAR(40) NOT NULL,
                    conflict_type VARCHAR(60) NOT NULL,
                    resolvability VARCHAR(60) NOT NULL,
                    one_line_summary TEXT NULL,
                    confidence DOUBLE NOT NULL,
                    retries INT NOT NULL,
                    stored_to_memory TINYINT(1) NOT NULL,
                    processing_time_seconds DOUBLE NULL,
                    trace_json JSON NULL,
                    created_at DATETIME NOT NULL,
                    UNIQUE KEY uq_user_request (user_id, request_id),
                    KEY idx_mediation_history_user_created (user_id, created_at),
                    CONSTRAINT fk_mediation_history_user
                        FOREIGN KEY (user_id) REFERENCES users(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_statuses (
                    id CHAR(36) PRIMARY KEY,
                    user_id CHAR(36) NOT NULL,
                    conversation_id VARCHAR(80) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'active',
                    latest_turn INT NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    resolved_at DATETIME NULL,
                    UNIQUE KEY uq_conversation_status_user_conv (user_id, conversation_id),
                    KEY idx_conversation_status_user_conv (user_id, conversation_id),
                    KEY idx_conversation_status_conv_id (conversation_id),
                    CONSTRAINT fk_conversation_status_user
                        FOREIGN KEY (user_id) REFERENCES users(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_resolutions (
                    id CHAR(36) PRIMARY KEY,
                    user_id CHAR(36) NOT NULL,
                    conversation_id VARCHAR(80) NOT NULL,
                    resolved_turn INT NOT NULL,
                    total_turns INT NOT NULL,
                    user_a_rating TINYINT NULL,
                    user_b_rating TINYINT NULL,
                    note TEXT NULL,
                    user_a_comment TEXT NULL,
                    user_b_comment TEXT NULL,
                    final_summary TEXT NULL,
                    conflict_type VARCHAR(60) NOT NULL,
                    resolvability VARCHAR(60) NOT NULL,
                    final_response_a TEXT NOT NULL,
                    final_response_b TEXT NOT NULL,
                    resolved_by_user_id CHAR(36) NOT NULL,
                    source VARCHAR(20) NOT NULL DEFAULT 'ui',
                    resolved_at DATETIME NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    UNIQUE KEY uq_conversation_resolution_user_conv (user_id, conversation_id),
                    KEY idx_conversation_resolution_user_conv (user_id, conversation_id),
                    KEY idx_conversation_resolution_conv_id (conversation_id),
                    KEY idx_conversation_resolution_resolved (user_id, resolved_at),
                    CONSTRAINT fk_conversation_resolution_user
                        FOREIGN KEY (user_id) REFERENCES users(id)
                        ON DELETE CASCADE,
                    CONSTRAINT fk_conversation_resolution_by_user
                        FOREIGN KEY (resolved_by_user_id) REFERENCES users(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            _ensure_resolution_feedback_schema(cursor)
        conn.commit()


def create_user(name: str, email: str, password: str) -> dict[str, Any]:
    connector = _mysql_connector()
    user_id = str(uuid.uuid4())
    salt = secrets.token_hex(16)
    now = _now()
    password_hash = _hash_password(password, salt)
    normalized_email = _normalize_email(email)

    with closing(_connect()) as conn:
        try:
            with closing(conn.cursor(dictionary=True)) as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (id, name, email, password_hash, password_salt, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (user_id, name.strip(), normalized_email, password_hash, salt, now, now),
                )
                cursor.execute("SELECT id, name, email, created_at FROM users WHERE id = %s", (user_id,))
                row = cursor.fetchone()
            conn.commit()
        except connector.IntegrityError as exc:
            conn.rollback()
            raise DuplicateUserError("Account already exists.") from exc

    if row is None:
        raise RuntimeError("User creation failed.")
    return _user_from_row(row)


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    with closing(_connect()) as conn:
        with closing(conn.cursor(dictionary=True)) as cursor:
            cursor.execute(
                """
                SELECT id, name, email, password_hash, password_salt, created_at
                FROM users
                WHERE email = %s
                """,
                (_normalize_email(email),),
            )
            row = cursor.fetchone()

    if row is None:
        return None

    password_hash = _hash_password(password, row["password_salt"])
    if not hmac.compare_digest(password_hash, row["password_hash"]):
        return None
    return _user_from_row(row)


def create_token(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = _now()
    with closing(_connect()) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                """
                INSERT INTO auth_tokens (token_hash, user_id, created_at, last_seen_at)
                VALUES (%s, %s, %s, %s)
                """,
                (_token_hash(token), user_id, now, now),
            )
        conn.commit()
    return token


def get_user_by_token(token: str) -> dict[str, Any] | None:
    token_digest = _token_hash(token)
    with closing(_connect()) as conn:
        with closing(conn.cursor(dictionary=True)) as cursor:
            cursor.execute(
                """
                SELECT users.id, users.name, users.email, users.created_at
                FROM auth_tokens
                JOIN users ON users.id = auth_tokens.user_id
                WHERE auth_tokens.token_hash = %s
                """,
                (token_digest,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cursor.execute(
                "UPDATE auth_tokens SET last_seen_at = %s WHERE token_hash = %s",
                (_now(), token_digest),
            )
        conn.commit()
    return _user_from_row(row)


def revoke_token(token: str) -> None:
    with closing(_connect()) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute("DELETE FROM auth_tokens WHERE token_hash = %s", (_token_hash(token),))
        conn.commit()


def _latest_history_turn(cursor, user_id: str, conversation_id: str) -> int:
    cursor.execute(
        """
        SELECT COALESCE(MAX(turn), 0) AS latest_turn
        FROM mediation_history
        WHERE user_id = %s AND conversation_id = %s
        """,
        (user_id, conversation_id),
    )
    row = cursor.fetchone() or {}
    return int(row.get("latest_turn") or 0)


def _resolution_payload(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "resolved": True,
        "conversation_id": row["conversation_id"],
        "resolved_turn": row["resolved_turn"],
        "user_a_rating": row["user_a_rating"],
        "user_b_rating": row["user_b_rating"],
        "note": row.get("note"),
        "user_a_comment": row.get("user_a_comment"),
        "user_b_comment": row.get("user_b_comment"),
        "rated": row.get("user_a_rating") is not None and row.get("user_b_rating") is not None,
        "resolved_at": _iso(row.get("resolved_at")),
        "conversation_status": "resolved",
        "final_summary": row.get("final_summary"),
        "conflict_type": row.get("conflict_type"),
        "resolvability": row.get("resolvability"),
    }


def get_resolution(user_id: str, conversation_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as conn:
        with closing(conn.cursor(dictionary=True)) as cursor:
            cursor.execute(
                """
                SELECT *
                FROM conversation_resolutions
                WHERE user_id = %s AND conversation_id = %s
                """,
                (user_id, conversation_id),
            )
            return _resolution_payload(cursor.fetchone())


def prepare_conversation_turn(user_id: str, conversation_id: str, turn: int) -> dict[str, Any]:
    """Validate the requested turn without holding locks during LLM work."""
    now = _now()
    with closing(_connect()) as conn:
        try:
            with closing(conn.cursor(dictionary=True)) as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversation_statuses (
                        id, user_id, conversation_id, status, latest_turn, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, 'active', 0, %s, %s)
                    ON DUPLICATE KEY UPDATE updated_at = updated_at
                    """,
                    (str(uuid.uuid4()), user_id, conversation_id, now, now),
                )
                cursor.execute(
                    """
                    SELECT *
                    FROM conversation_statuses
                    WHERE user_id = %s AND conversation_id = %s
                    FOR UPDATE
                    """,
                    (user_id, conversation_id),
                )
                row = cursor.fetchone()
                if row is None:
                    conn.rollback()
                    return {"ok": False, "error": "CONVERSATION_STATE_ERROR", "status": "unknown"}

                if row["latest_turn"] == 0:
                    historical_turn = _latest_history_turn(cursor, user_id, conversation_id)
                    if historical_turn:
                        cursor.execute(
                            """
                            UPDATE conversation_statuses
                            SET latest_turn = %s, updated_at = %s
                            WHERE user_id = %s AND conversation_id = %s
                            """,
                            (historical_turn, now, user_id, conversation_id),
                        )
                        row["latest_turn"] = historical_turn

                if row["status"] == "resolved":
                    cursor.execute(
                        """
                        SELECT *
                        FROM conversation_resolutions
                        WHERE user_id = %s AND conversation_id = %s
                        """,
                        (user_id, conversation_id),
                    )
                    resolution = _resolution_payload(cursor.fetchone())
                    conn.commit()
                    return {
                        "ok": False,
                        "error": "ALREADY_RESOLVED",
                        "status": "resolved",
                        **(resolution or {}),
                    }

                latest_turn = int(row["latest_turn"])
                expected_turn = latest_turn + 1
                if turn != expected_turn:
                    conn.commit()
                    return {
                        "ok": False,
                        "error": "STALE_TURN",
                        "status": row["status"],
                        "latest_turn": latest_turn,
                        "expected_turn": expected_turn,
                    }

            conn.commit()
            return {"ok": True, "status": "active", "latest_turn": latest_turn, "expected_turn": expected_turn}
        except Exception:
            conn.rollback()
            raise


def save_mediation_turn(
    *,
    user_id: str,
    mode: str,
    text_a: str,
    text_b: str,
    response: Any,
) -> dict[str, Any]:
    """Atomically save the completed turn and advance conversation state."""
    trace = response.trace or {}
    trace_json = json.dumps(trace, ensure_ascii=False)
    now = _now()
    with closing(_connect()) as conn:
        try:
            with closing(conn.cursor(dictionary=True)) as cursor:
                cursor.execute(
                    """
                    SELECT *
                    FROM conversation_statuses
                    WHERE user_id = %s AND conversation_id = %s
                    FOR UPDATE
                    """,
                    (user_id, response.conversation_id),
                )
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(
                        """
                        INSERT INTO conversation_statuses (
                            id, user_id, conversation_id, status, latest_turn, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, 'active', 0, %s, %s)
                        """,
                        (str(uuid.uuid4()), user_id, response.conversation_id, now, now),
                    )
                    row = {"status": "active", "latest_turn": 0}

                if row["status"] == "resolved":
                    cursor.execute(
                        """
                        SELECT *
                        FROM conversation_resolutions
                        WHERE user_id = %s AND conversation_id = %s
                        """,
                        (user_id, response.conversation_id),
                    )
                    resolution = _resolution_payload(cursor.fetchone())
                    conn.rollback()
                    return {
                        "ok": False,
                        "error": "ALREADY_RESOLVED",
                        "status": "resolved",
                        **(resolution or {}),
                    }

                latest_turn = int(row["latest_turn"])
                expected_turn = latest_turn + 1
                if response.turn != expected_turn:
                    conn.rollback()
                    return {
                        "ok": False,
                        "error": "STALE_TURN",
                        "status": row["status"],
                        "latest_turn": latest_turn,
                        "expected_turn": expected_turn,
                    }

                cursor.execute(
                    """
                    INSERT INTO mediation_history (
                        id,
                        user_id,
                        conversation_id,
                        turn,
                        mode,
                        request_id,
                        trace_id,
                        text_a,
                        text_b,
                        response_a,
                        response_b,
                        conversation_status,
                        conflict_type,
                        resolvability,
                        one_line_summary,
                        confidence,
                        retries,
                        stored_to_memory,
                        processing_time_seconds,
                        trace_json,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        user_id,
                        response.conversation_id,
                        response.turn,
                        mode,
                        response.request_id,
                        response.trace_id,
                        text_a,
                        text_b,
                        response.response_a,
                        response.response_b,
                        response.conversation_status,
                        response.conflict_type,
                        response.resolvability,
                        response.one_line_summary,
                        response.confidence,
                        response.retries,
                        1 if response.stored_to_memory else 0,
                        response.processing_time_seconds,
                        trace_json,
                        now,
                    ),
                )
                cursor.execute(
                    """
                    UPDATE conversation_statuses
                    SET latest_turn = %s, updated_at = %s
                    WHERE user_id = %s AND conversation_id = %s
                    """,
                    (response.turn, now, user_id, response.conversation_id),
                )
            conn.commit()
            return {"ok": True, "status": "active", "latest_turn": response.turn}
        except Exception:
            conn.rollback()
            raise


def save_mediation_history(
    *,
    user_id: str,
    mode: str,
    text_a: str,
    text_b: str,
    response: Any,
) -> None:
    trace = response.trace or {}
    trace_json = json.dumps(trace, ensure_ascii=False)
    now = _now()
    with closing(_connect()) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                """
                INSERT IGNORE INTO mediation_history (
                    id,
                    user_id,
                    conversation_id,
                    turn,
                    mode,
                    request_id,
                    trace_id,
                    text_a,
                    text_b,
                    response_a,
                    response_b,
                    conversation_status,
                    conflict_type,
                    resolvability,
                    one_line_summary,
                    confidence,
                    retries,
                    stored_to_memory,
                    processing_time_seconds,
                    trace_json,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    user_id,
                    response.conversation_id,
                    response.turn,
                    mode,
                    response.request_id,
                    response.trace_id,
                    text_a,
                    text_b,
                    response.response_a,
                    response.response_b,
                    response.conversation_status,
                    response.conflict_type,
                    response.resolvability,
                    response.one_line_summary,
                    response.confidence,
                    response.retries,
                    1 if response.stored_to_memory else 0,
                    response.processing_time_seconds,
                    trace_json,
                    now,
                ),
            )
        conn.commit()


def resolve_conversation(
    *,
    user_id: str,
    conversation_id: str,
    user_a_rating: int | None = None,
    user_b_rating: int | None = None,
    note: str | None = None,
    user_a_comment: str | None = None,
    user_b_comment: str | None = None,
    source: str = "ui",
) -> dict[str, Any]:
    now = _now()
    feedback_supplied = any(
        value is not None
        for value in (user_a_rating, user_b_rating, note, user_a_comment, user_b_comment)
    )
    with closing(_connect()) as conn:
        try:
            with closing(conn.cursor(dictionary=True)) as cursor:
                cursor.execute(
                    """
                    SELECT *
                    FROM conversation_statuses
                    WHERE user_id = %s AND conversation_id = %s
                    FOR UPDATE
                    """,
                    (user_id, conversation_id),
                )
                status = cursor.fetchone()
                if status is None:
                    latest_turn = _latest_history_turn(cursor, user_id, conversation_id)
                    if latest_turn <= 0:
                        conn.rollback()
                        return {
                            "ok": False,
                            "error": "CONVERSATION_NOT_FOUND",
                            "status": "missing",
                        }
                    cursor.execute(
                        """
                        INSERT INTO conversation_statuses (
                            id, user_id, conversation_id, status, latest_turn, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, 'active', %s, %s, %s)
                        """,
                        (str(uuid.uuid4()), user_id, conversation_id, latest_turn, now, now),
                    )
                    cursor.execute(
                        """
                        SELECT *
                        FROM conversation_statuses
                        WHERE user_id = %s AND conversation_id = %s
                        FOR UPDATE
                        """,
                        (user_id, conversation_id),
                    )
                    status = cursor.fetchone()

                if status and status["status"] == "resolved":
                    if feedback_supplied:
                        cursor.execute(
                            """
                            UPDATE conversation_resolutions
                            SET user_a_rating = COALESCE(%s, user_a_rating),
                                user_b_rating = COALESCE(%s, user_b_rating),
                                note = COALESCE(%s, note),
                                user_a_comment = COALESCE(%s, user_a_comment),
                                user_b_comment = COALESCE(%s, user_b_comment),
                                source = %s,
                                updated_at = %s
                            WHERE user_id = %s AND conversation_id = %s
                            """,
                            (
                                user_a_rating,
                                user_b_rating,
                                note,
                                user_a_comment,
                                user_b_comment,
                                source,
                                now,
                                user_id,
                                conversation_id,
                            ),
                        )
                    cursor.execute(
                        """
                        SELECT *
                        FROM conversation_resolutions
                        WHERE user_id = %s AND conversation_id = %s
                        """,
                        (user_id, conversation_id),
                    )
                    resolution = _resolution_payload(cursor.fetchone())
                    conn.commit()
                    return {"ok": True, **(resolution or {"resolved": True, "conversation_id": conversation_id})}

                cursor.execute(
                    """
                    SELECT *
                    FROM mediation_history
                    WHERE user_id = %s AND conversation_id = %s
                    ORDER BY turn DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    (user_id, conversation_id),
                )
                last = cursor.fetchone()
                if last is None:
                    conn.rollback()
                    return {
                        "ok": False,
                        "error": "CONVERSATION_NOT_FOUND",
                        "status": "missing",
                    }

                cursor.execute(
                    """
                    INSERT INTO conversation_resolutions (
                        id,
                        user_id,
                        conversation_id,
                        resolved_turn,
                        total_turns,
                        user_a_rating,
                        user_b_rating,
                        note,
                        user_a_comment,
                        user_b_comment,
                        final_summary,
                        conflict_type,
                        resolvability,
                        final_response_a,
                        final_response_b,
                        resolved_by_user_id,
                        source,
                        resolved_at,
                        created_at,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        user_a_rating = COALESCE(VALUES(user_a_rating), user_a_rating),
                        user_b_rating = COALESCE(VALUES(user_b_rating), user_b_rating),
                        note = COALESCE(VALUES(note), note),
                        user_a_comment = COALESCE(VALUES(user_a_comment), user_a_comment),
                        user_b_comment = COALESCE(VALUES(user_b_comment), user_b_comment),
                        final_summary = VALUES(final_summary),
                        conflict_type = VALUES(conflict_type),
                        resolvability = VALUES(resolvability),
                        final_response_a = VALUES(final_response_a),
                        final_response_b = VALUES(final_response_b),
                        total_turns = VALUES(total_turns),
                        resolved_by_user_id = VALUES(resolved_by_user_id),
                        source = VALUES(source),
                        updated_at = VALUES(updated_at)
                    """,
                    (
                        str(uuid.uuid4()),
                        user_id,
                        conversation_id,
                        last["turn"],
                        last["turn"],
                        user_a_rating,
                        user_b_rating,
                        note,
                        user_a_comment,
                        user_b_comment,
                        last.get("one_line_summary"),
                        last["conflict_type"],
                        last["resolvability"],
                        last["response_a"],
                        last["response_b"],
                        user_id,
                        source,
                        now,
                        now,
                        now,
                    ),
                )
                cursor.execute(
                    """
                    UPDATE conversation_statuses
                    SET status = 'resolved',
                        latest_turn = %s,
                        resolved_at = COALESCE(resolved_at, %s),
                        updated_at = %s
                    WHERE user_id = %s AND conversation_id = %s
                    """,
                    (last["turn"], now, now, user_id, conversation_id),
                )
                cursor.execute(
                    """
                    SELECT *
                    FROM conversation_resolutions
                    WHERE user_id = %s AND conversation_id = %s
                    """,
                    (user_id, conversation_id),
                )
                resolution = _resolution_payload(cursor.fetchone())
            conn.commit()
            return {"ok": True, **(resolution or {})}
        except Exception:
            conn.rollback()
            raise


def _history_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    trace: dict[str, Any] = {}
    if row.get("safety_score") is not None:
        trace["safety_score"] = float(row["safety_score"])
    if row.get("rag_used") is not None:
        trace["rag_used"] = bool(row["rag_used"])
    if row.get("retrieved_cases") is not None:
        trace["retrieved_cases"] = int(row["retrieved_cases"])
    if row.get("intent_a"):
        trace["intent_a"] = row["intent_a"]
    if row.get("intent_b"):
        trace["intent_b"] = row["intent_b"]
    if row.get("intent_confidence_a") is not None:
        trace["intent_confidence_a"] = float(row["intent_confidence_a"])
    if row.get("intent_confidence_b") is not None:
        trace["intent_confidence_b"] = float(row["intent_confidence_b"])
    created_at = _iso(row["created_at"])
    resolved_at = _iso(row.get("resolution_resolved_at") or row.get("status_resolved_at"))
    lifecycle_status = row.get("lifecycle_status") or "active"
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "conversation_id": row["conversation_id"],
        "turn": row["turn"],
        "mode": row["mode"],
        "request_id": row["request_id"],
        "trace_id": row["trace_id"],
        "response_a": row["response_a"],
        "response_b": row["response_b"],
        "conversation_status": row["conversation_status"],
        "conflict_type": row["conflict_type"],
        "resolvability": row["resolvability"],
        "one_line_summary": row["one_line_summary"],
        "confidence": row["confidence"],
        "retries": row["retries"],
        "stored_to_memory": bool(row["stored_to_memory"]),
        "processing_time_seconds": row["processing_time_seconds"],
        "trace": trace,
        "created_at": created_at,
        "conversation_lifecycle_status": lifecycle_status,
        "latest_turn": row.get("latest_turn") or row["turn"],
        "resolved": bool(row.get("resolved_turn") or lifecycle_status == "resolved"),
        "resolved_at": resolved_at,
        "resolved_turn": row.get("resolved_turn"),
        "user_a_rating": row.get("user_a_rating"),
        "user_b_rating": row.get("user_b_rating"),
        "resolution_note": row.get("resolution_note"),
        "user_a_comment": row.get("user_a_comment"),
        "user_b_comment": row.get("user_b_comment"),
        "rated": row.get("user_a_rating") is not None and row.get("user_b_rating") is not None,
    }


def list_mediation_history(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    bounded_limit = min(max(limit, 1), 200)
    with closing(_connect()) as conn:
        with closing(conn.cursor(dictionary=True)) as cursor:
            cursor.execute(
                """
                SELECT
                    mediation_history.id,
                    mediation_history.user_id,
                    mediation_history.conversation_id,
                    mediation_history.turn,
                    mediation_history.mode,
                    mediation_history.request_id,
                    mediation_history.trace_id,
                    mediation_history.response_a,
                    mediation_history.response_b,
                    mediation_history.conversation_status,
                    mediation_history.conflict_type,
                    mediation_history.resolvability,
                    mediation_history.one_line_summary,
                    mediation_history.confidence,
                    mediation_history.retries,
                    mediation_history.stored_to_memory,
                    mediation_history.processing_time_seconds,
                    mediation_history.created_at,
                    CAST(JSON_UNQUOTE(JSON_EXTRACT(mediation_history.trace_json, '$.safety_score')) AS DECIMAL(8,4))
                        AS safety_score,
                    CASE JSON_UNQUOTE(JSON_EXTRACT(mediation_history.trace_json, '$.rag_used'))
                        WHEN 'true' THEN 1
                        WHEN '1' THEN 1
                        ELSE 0
                    END AS rag_used,
                    CAST(JSON_UNQUOTE(JSON_EXTRACT(mediation_history.trace_json, '$.retrieved_cases')) AS UNSIGNED)
                        AS retrieved_cases,
                    JSON_UNQUOTE(JSON_EXTRACT(mediation_history.trace_json, '$.intent_a'))
                        AS intent_a,
                    JSON_UNQUOTE(JSON_EXTRACT(mediation_history.trace_json, '$.intent_b'))
                        AS intent_b,
                    CAST(JSON_UNQUOTE(JSON_EXTRACT(mediation_history.trace_json, '$.intent_confidence_a')) AS DECIMAL(8,4))
                        AS intent_confidence_a,
                    CAST(JSON_UNQUOTE(JSON_EXTRACT(mediation_history.trace_json, '$.intent_confidence_b')) AS DECIMAL(8,4))
                        AS intent_confidence_b,
                    conversation_statuses.status AS lifecycle_status,
                    conversation_statuses.latest_turn AS latest_turn,
                    conversation_statuses.resolved_at AS status_resolved_at,
                    conversation_resolutions.resolved_turn AS resolved_turn,
                    conversation_resolutions.user_a_rating AS user_a_rating,
                    conversation_resolutions.user_b_rating AS user_b_rating,
                    conversation_resolutions.note AS resolution_note,
                    conversation_resolutions.user_a_comment AS user_a_comment,
                    conversation_resolutions.user_b_comment AS user_b_comment,
                    conversation_resolutions.resolved_at AS resolution_resolved_at
                FROM mediation_history
                LEFT JOIN conversation_statuses
                    ON conversation_statuses.user_id = mediation_history.user_id
                    AND conversation_statuses.conversation_id = mediation_history.conversation_id
                LEFT JOIN conversation_resolutions
                    ON conversation_resolutions.user_id = mediation_history.user_id
                    AND conversation_resolutions.conversation_id = mediation_history.conversation_id
                WHERE mediation_history.user_id = %s
                ORDER BY mediation_history.created_at DESC
                LIMIT %s
                """,
                (user_id, bounded_limit),
            )
            rows = cursor.fetchall()
    return [_history_from_row(row) for row in rows]
