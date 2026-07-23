"""Runtime FTS-corruption self-heal on the SessionDB write path."""

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


def _corrupt_fts(db_path, table="messages_fts"):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        f"UPDATE {table}_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _message_contents(db_path):
    raw = sqlite3.connect(str(db_path))
    rows = raw.execute("SELECT content FROM messages ORDER BY id").fetchall()
    raw.close()
    return [r[0] for r in rows]


class TestRuntimeFtsRebuild:
    def test_corruption_error_classification_covers_both_sqlite_messages(self):
        assert SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError("database disk image is malformed")
        )
        assert SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError(
                'fts5: corrupt structure record for table "messages_fts"'
            )
        )
        assert not SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError("no such table: nothing_fts_related")
        )

    def test_append_self_heals_after_fts_corruption(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "hello world")
        _corrupt_fts(tmp_path / "state.db")

        msg_id = db.append_message("s1", "user", "healed append")

        assert msg_id is not None
        assert _message_contents(tmp_path / "state.db") == [
            "hello world",
            "healed append",
        ]

    def test_search_works_after_self_heal(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "before corruption")
        _corrupt_fts(tmp_path / "state.db")
        db.append_message("s1", "user", "searchable needle text")

        raw = sqlite3.connect(str(tmp_path / "state.db"))
        hits = raw.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'needle'"
        ).fetchall()
        raw.close()
        assert len(hits) == 1

    def test_append_self_heals_after_trigram_corruption(self, db, tmp_path):
        if not db._trigram_available:
            pytest.skip("trigram FTS unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "记忆断裂之前")
        _corrupt_fts(tmp_path / "state.db", "messages_fts_trigram")

        msg_id = db.append_message("s1", "user", "记忆断裂之后")

        assert msg_id is not None
        check = sqlite3.connect(str(tmp_path / "state.db"))
        check.execute(
            "INSERT INTO messages_fts_trigram(messages_fts_trigram, rank) "
            "VALUES('integrity-check', 1)"
        ).fetchall()
        check.rollback()
        check.close()

    def test_disable_trigram_drops_corrupt_index_without_touching_messages(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"
        seeded = SessionDB(db_path=db_path)
        if not seeded._trigram_available:
            seeded.close()
            pytest.skip("trigram FTS unavailable in this build")
        seeded.create_session("s1", source="test")
        seeded.append_message("s1", "user", "canonical message")
        seeded.close()
        _corrupt_fts(db_path, "messages_fts_trigram")

        monkeypatch.setenv("HERMES_DISABLE_FTS_TRIGRAM", "1")
        disabled = SessionDB(db_path=db_path)
        try:
            assert _message_contents(db_path) == ["canonical message"]
            assert disabled._fts_table_exists("messages_fts") is True
            assert disabled._fts_table_exists("messages_fts_trigram") is False
            check = sqlite3.connect(str(db_path))
            check.execute(
                "INSERT INTO messages_fts(messages_fts, rank) "
                "VALUES('integrity-check', 1)"
            ).fetchall()
            check.rollback()
            check.close()
        finally:
            disabled.close()

    def test_rebuild_is_one_shot_per_instance(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(tmp_path / "state.db")
        db.append_message("s1", "user", "first heal")
        assert db._fts_runtime_rebuild_attempted is True

        _corrupt_fts(tmp_path / "state.db")
        with pytest.raises(sqlite3.DatabaseError):
            db.append_message("s1", "user", "second corruption")

    def test_non_fts_errors_still_propagate(self, db):
        db.create_session("s1", source="test")

        def _bad(conn):
            raise sqlite3.IntegrityError("NOT NULL constraint failed: x.y")

        with pytest.raises(sqlite3.IntegrityError):
            db._execute_write(_bad)
        assert db._fts_runtime_rebuild_attempted is False

    def test_lock_retry_path_unchanged(self, db):
        calls = {"n": 0}

        def _flaky(conn):
            calls["n"] += 1
            if calls["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        assert db._execute_write(_flaky) == "ok"
        assert calls["n"] == 3
        assert db._fts_runtime_rebuild_attempted is False
