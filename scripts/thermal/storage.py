"""Transactional resumable predictions and disk-backed search pools."""

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3

from .common import chunks, finite, write_csv


@contextmanager
def output_lock(directory):
    """OS-released lock: interruption never leaves a stale lock blocking resume."""
    path = Path(directory) / ".run.lock"
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another process is using {directory}") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another process is using {directory}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def group_key(group):
    return ";".join(sorted(group, key=lambda m: (int(m[1:-1]), m[-1])))


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                key TEXT PRIMARY KEY, mutation_count INTEGER NOT NULL,
                score REAL, status TEXT NOT NULL, error TEXT
            );
            CREATE TABLE IF NOT EXISTS pool (
                depth INTEGER NOT NULL, key TEXT NOT NULL,
                score REAL NOT NULL, ratio REAL NOT NULL,
                parent TEXT, marginal REAL,
                PRIMARY KEY (depth, key)
            );
            CREATE INDEX IF NOT EXISTS pool_order ON pool(depth, score, key);
            CREATE TABLE IF NOT EXISTS extensions (
                key TEXT PRIMARY KEY, parent TEXT NOT NULL, parent_score REAL NOT NULL
            );
        """)

    def close(self):
        self.db.close()

    def export_failures(self, path):
        fields = ["mutation_model", "mutation_count", "model_kind", "status", "error"]
        write_csv(path, fields, (
            {"mutation_model": key.replace(";", "/"), "mutation_count": count,
             "model_kind": "multi", "status": status, "error": error}
            for key, count, status, error in self.db.execute(
                "SELECT key,mutation_count,status,error FROM predictions WHERE status!='ok' ORDER BY key"
            )
        ))

    def reset_pool(self, depth):
        with self.db:
            self.db.execute("DELETE FROM pool WHERE depth = ?", (depth,))

    def add_pool(self, rows):
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO pool VALUES (?, ?, ?, ?, ?, ?)", rows)

    def ordered_pool(self, depth, high_ratio, high):
        operator = ">=" if high else "<"
        yield from self.db.execute(
            f"SELECT key,score,ratio,parent,marginal FROM pool WHERE depth=? AND ratio {operator} ? ORDER BY score,key",
            (depth, high_ratio),
        )

    def pool_count(self, depth):
        return self.db.execute("SELECT count(*) FROM pool WHERE depth=?", (depth,)).fetchone()[0]

    def reset_extensions(self):
        with self.db:
            self.db.execute("DELETE FROM extensions")

    def add_extensions(self, rows):
        # Any beam parent may justify a child. The largest parent score gives
        # the most negative child-parent marginal; retain that witness.
        with self.db:
            self.db.executemany("""
                INSERT INTO extensions VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET parent=excluded.parent, parent_score=excluded.parent_score
                WHERE excluded.parent_score > extensions.parent_score
                   OR (excluded.parent_score = extensions.parent_score AND excluded.parent < extensions.parent)
            """, rows)

    def extensions(self):
        yield from self.db.execute("SELECT key,parent,parent_score FROM extensions ORDER BY key")


class CachedScorer:
    def __init__(self, store, backend, batch_size):
        self.store, self.backend, self.batch_size = store, backend, batch_size
        self.new_count, self.hit_count = 0, 0

    def score(self, groups):
        keys = [group_key(group) for group in groups]
        if len({len(group) for group in groups}) > 1:
            raise ValueError("Cannot mix mutation counts in a cache batch")
        cached = {}
        for part in chunks(dict.fromkeys(keys), 400):
            placeholders = ",".join("?" for _ in part)
            cached.update(self.store.db.execute(
                f"SELECT key,score FROM predictions WHERE status='ok' AND key IN ({placeholders})", part
            ))
        self.hit_count += sum(key in cached for key in keys)
        missing = [key for key in dict.fromkeys(keys) if key not in cached]
        for part in chunks(missing, self.batch_size):
            try:
                scores = self.backend.score([key.split(";") for key in part])
                if len(scores) != len(part):
                    raise ValueError("Backend returned wrong number of scores")
                scores = [finite(score) for score in scores]
            except Exception as exc:
                with self.store.db:
                    self.store.db.executemany("INSERT OR REPLACE INTO predictions VALUES (?,?,NULL,'error',?)", (
                        (key, len(key.split(";")), f"{type(exc).__name__}: {exc}") for key in part
                    ))
                raise  # An incomplete graph/search must never look successful.
            with self.store.db:
                self.store.db.executemany("INSERT OR REPLACE INTO predictions VALUES (?,?,?,'ok',NULL)", (
                    (key, len(key.split(";")), score) for key, score in zip(part, scores)
                ))
            cached.update(zip(part, scores))
            self.new_count += len(part)
        return [finite(cached[key]) for key in keys]
