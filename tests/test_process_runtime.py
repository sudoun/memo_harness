import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path

from posterior_memory_harness import (
    CURRENT_SCHEMA_VERSION,
    CalibrationDriftMonitor,
    PosteriorMemoryHarness,
    SQLiteMemoryStore,
    cyclic_law,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def child_environment() -> dict[str, str]:
    """Make source-tree subprocesses behave like an installed CLI."""
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SOURCE_ROOT)
        if not current
        else str(SOURCE_ROOT) + os.pathsep + current
    )
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def run_cli(*arguments: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "posterior_memory_harness", *arguments],
        cwd=PROJECT_ROOT,
        env=child_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


CONCURRENT_OPEN_WORKER = """
import json
import sys
import time
from pathlib import Path

from posterior_memory_harness import SQLiteMemoryStore

database = Path(sys.argv[1])
ready = Path(sys.argv[2])
gate = Path(sys.argv[3])
ready.write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 20.0
while not gate.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("parent did not release concurrent-open gate")
    time.sleep(0.005)
status = SQLiteMemoryStore(database).schema_status()
with SQLiteMemoryStore(database, initialize=False)._connection() as connection:
    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
print(json.dumps({
    "version": status.version,
    "migrations": [item.version for item in status.migrations],
    "journal_mode": journal_mode,
}))
"""

WAL_RETRY_WORKER = """
import json
import sqlite3
import sys
import time
from pathlib import Path

from posterior_memory_harness import SQLiteMemoryStore

database = Path(sys.argv[1])
ready = Path(sys.argv[2])
gate = Path(sys.argv[3])
store = SQLiteMemoryStore(database, initialize=False)
ready.write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 20.0
while not gate.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("parent did not release WAL retry gate")
    time.sleep(0.005)
store._enable_wal()
with sqlite3.connect(database) as connection:
    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
print(json.dumps({"journal_mode": journal_mode}))
"""

ABORTED_APPEND_WORKER = """
import json
import sys
from pathlib import Path

from posterior_memory_harness import (
    CalibrationDriftMonitor,
    PosteriorMemoryHarness,
    SQLiteMemoryStore,
    cyclic_law,
)

database = Path(sys.argv[1])
payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
store = SQLiteMemoryStore(database, initialize=False)
harness = PosteriorMemoryHarness(
    store,
    monitor=CalibrationDriftMonitor(),
).register_relation("phase", cyclic_law(2))
harness.observe_payload(payload)
"""


class CrossProcessRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.db = self.root / "memory.sqlite"

    def tearDown(self):
        self.temporary.cleanup()

    def _write_observation(self, identifier: str = "process-observation") -> Path:
        path = self.root / f"{identifier}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "observation_id": identifier,
                    "namespace": "process-test",
                    "relation_type": "phase",
                    "source": "a",
                    "target": "b",
                    "posterior": [0.2, 0.8],
                    "prior": [0.5, 0.5],
                    "evidence_family": "process-family",
                    "observed_at": 1.0,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_two_processes_concurrently_migrate_the_same_new_database(self):
        gate = self.root / "start"
        processes: list[subprocess.Popen] = []
        ready_files: list[Path] = []
        for index in range(2):
            ready = self.root / f"ready-{index}"
            ready_files.append(ready)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        CONCURRENT_OPEN_WORKER,
                        str(self.db),
                        str(ready),
                        str(gate),
                    ],
                    cwd=PROJECT_ROOT,
                    env=child_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            )

        deadline = time.monotonic() + 15.0
        while not all(path.exists() for path in ready_files):
            failed = [
                process.returncode
                for process in processes
                if process.poll() is not None
            ]
            if failed:
                outputs = [process.communicate() for process in processes]
                self.fail(
                    f"migration worker exited before the gate: "
                    f"{failed!r}; outputs={outputs!r}"
                )
            if time.monotonic() >= deadline:
                for process in processes:
                    process.kill()
                self.fail("migration workers did not reach the start gate")
            time.sleep(0.01)

        gate.write_text("go", encoding="utf-8")
        payloads = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=30.0)
            self.assertEqual(
                process.returncode,
                0,
                msg=f"stdout={stdout!r}; stderr={stderr!r}",
            )
            payloads.append(json.loads(stdout))

        self.assertEqual(
            payloads,
            [
                {
                    "version": CURRENT_SCHEMA_VERSION,
                    "migrations": [1, 2, 3],
                    "journal_mode": "wal",
                },
                {
                    "version": CURRENT_SCHEMA_VERSION,
                    "migrations": [1, 2, 3],
                    "journal_mode": "wal",
                },
            ],
        )
        with closing(sqlite3.connect(self.db)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            migrations = connection.execute(
                """
                SELECT version, COUNT(*)
                FROM memory_schema_migrations
                GROUP BY version
                ORDER BY version
                """
            ).fetchall()
            quick_check = connection.execute(
                "PRAGMA quick_check"
            ).fetchone()[0]
        self.assertEqual(version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(migrations, [(1, 1), (2, 1), (3, 1)])
        self.assertEqual(quick_check, "ok")

    def test_prediction_trigger_abort_rolls_back_observation_across_process(self):
        store = SQLiteMemoryStore(self.db)
        PosteriorMemoryHarness(
            store,
            monitor=CalibrationDriftMonitor(),
        ).register_relation("phase", cyclic_law(2))
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute(
                """
                CREATE TRIGGER process_abort_prediction
                BEFORE INSERT ON monitor_predictions
                BEGIN
                    SELECT RAISE(ABORT, 'cross-process failpoint');
                END
                """
            )

        payload = self._write_observation("aborted")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                ABORTED_APPEND_WORKER,
                str(self.db),
                str(payload),
            ],
            cwd=PROJECT_ROOT,
            env=child_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30.0,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("transaction", result.stderr)

        with closing(sqlite3.connect(self.db)) as connection:
            counts = (
                connection.execute(
                    "SELECT COUNT(*) FROM observations"
                ).fetchone()[0],
                connection.execute(
                    "SELECT COUNT(*) FROM monitor_predictions"
                ).fetchone()[0],
            )
            quick_check = connection.execute(
                "PRAGMA quick_check"
            ).fetchone()[0]
        self.assertEqual(counts, (0, 0))
        self.assertEqual(quick_check, "ok")

    def test_wal_transition_retries_until_cross_process_reader_releases(self):
        SQLiteMemoryStore(self.db)
        with closing(sqlite3.connect(self.db)) as connection:
            journal_mode = connection.execute(
                "PRAGMA journal_mode=DELETE"
            ).fetchone()[0]
        self.assertEqual(journal_mode, "delete")

        ready = self.root / "wal-ready"
        gate = self.root / "wal-start"
        with closing(sqlite3.connect(self.db)) as blocker:
            blocker.execute("BEGIN")
            blocker.execute("SELECT COUNT(*) FROM observations").fetchone()
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    WAL_RETRY_WORKER,
                    str(self.db),
                    str(ready),
                    str(gate),
                ],
                cwd=PROJECT_ROOT,
                env=child_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            deadline = time.monotonic() + 15.0
            while not ready.exists():
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    self.fail(
                        "WAL retry worker exited before the gate: "
                        f"stdout={stdout!r}; stderr={stderr!r}"
                    )
                if time.monotonic() >= deadline:
                    process.kill()
                    self.fail("WAL retry worker did not reach the gate")
                time.sleep(0.01)
            gate.write_text("go", encoding="utf-8")
            time.sleep(0.2)
            self.assertIsNone(
                process.poll(),
                "WAL transition unexpectedly bypassed the active reader",
            )
            blocker.rollback()

        stdout, stderr = process.communicate(timeout=30.0)
        self.assertEqual(
            process.returncode,
            0,
            msg=f"stdout={stdout!r}; stderr={stderr!r}",
        )
        self.assertEqual(json.loads(stdout), {"journal_mode": "wal"})

    def test_module_cli_restarts_from_sqlite_and_preserves_exit_codes(self):
        payload = self._write_observation("restart")
        base = (
            "--db",
            str(self.db),
            "--relation-type",
            "phase",
            "--law",
            "cyclic:2",
        )
        first = run_cli(*base, "observe", str(payload))
        self.assertEqual(
            first.returncode,
            0,
            msg=f"stdout={first.stdout!r}; stderr={first.stderr!r}",
        )
        self.assertEqual(json.loads(first.stdout)["observation_id"], "restart")

        restarted = run_cli(
            "--db",
            str(self.db),
            "health",
            "--namespace",
            "process-test",
        )
        self.assertEqual(
            restarted.returncode,
            0,
            msg=f"stdout={restarted.stdout!r}; stderr={restarted.stderr!r}",
        )
        overall = json.loads(restarted.stdout)["relations"]["phase"]["overall"]
        self.assertEqual(overall["counts"]["tracked"], 1)

        duplicate = run_cli(*base, "observe", str(payload))
        self.assertEqual(
            duplicate.returncode,
            4,
            msg=f"stdout={duplicate.stdout!r}; stderr={duplicate.stderr!r}",
        )
        self.assertEqual(
            json.loads(duplicate.stderr)["error"]["code"],
            "conflict",
        )


if __name__ == "__main__":
    unittest.main()
