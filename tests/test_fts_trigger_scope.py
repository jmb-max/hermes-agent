"""FTS update triggers must ignore metadata-only message updates."""

import sqlite3

import pytest

from hermes_state import SessionDB


def _corrupt_base_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def test_metadata_update_does_not_touch_fts_but_content_update_does(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        message_id = db.append_message("s1", "user", "before")
        _corrupt_base_fts(db_path)

        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET active = 0 WHERE id = ?", (message_id,)
            )
        )
        assert db._fts_runtime_rebuild_attempted is False

        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                ("after", message_id),
            )
        )
        assert db._fts_runtime_rebuild_attempted is True
        assert db._conn.execute(
            "SELECT content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()[0] == "after"
    finally:
        try:
            db.close()
        except Exception:
            pass


def test_reopen_replaces_legacy_unscoped_update_triggers(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    if not db._fts_enabled:
        db.close()
        pytest.skip("FTS5 unavailable in this build")
    db.close()

    raw = sqlite3.connect(str(db_path))
    for name, table in (
        ("messages_fts_update", "messages_fts"),
        ("messages_fts_trigram_update", "messages_fts_trigram"),
    ):
        raw.execute(f"DROP TRIGGER {name}")
        raw.executescript(
            f"""
            CREATE TRIGGER {name} AFTER UPDATE ON messages
            WHEN old.content IS NOT new.content
            BEGIN
                DELETE FROM {table} WHERE rowid = old.id;
                INSERT INTO {table}(rowid, content) VALUES (
                    new.id,
                    COALESCE(new.content, '') || ' ' ||
                    COALESCE(new.tool_name, '') || ' ' ||
                    COALESCE(new.tool_calls, '')
                );
            END;
            """
        )
    raw.commit()
    raw.close()

    reopened = SessionDB(db_path=db_path)
    try:
        rows = reopened._conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE name IN ('messages_fts_update', 'messages_fts_trigram_update')"
        ).fetchall()
        assert len(rows) == 2
        for row in rows:
            sql = row["sql"].lower()
            assert "old.content is not new.content" in sql
            assert "old.tool_name is not new.tool_name" in sql
            assert "old.tool_calls is not new.tool_calls" in sql
    finally:
        reopened.close()
