import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from posterior_memory_harness import CURRENT_SCHEMA_VERSION
from posterior_memory_harness.cli import main, schema_main


class DurableCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.db = root / "memory.sqlite"
        self.observation = root / "observation.json"
        self.query = root / "query.json"
        self.outcome = root / "outcome.json"
        self.observation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "observation_id": "obs-1",
                    "namespace": "run",
                    "relation_type": "phase",
                    "source": "a",
                    "target": "b",
                    "posterior": [0.1, 0.8, 0.1],
                    "prior": [1 / 3, 1 / 3, 1 / 3],
                    "evidence_family": "event-1",
                    "observed_at": 1.0,
                }
            ),
            encoding="utf-8",
        )
        self.query.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "namespace": "run",
                    "relation_type": "phase",
                    "nodes": ["a", "b"],
                    "root": "a",
                    "as_of": 2.0,
                    "top_k": 3,
                }
            ),
            encoding="utf-8",
        )
        self.outcome.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "observation_id": "obs-1",
                    "true_relation": "1",
                    "outcome_event_id": "outcome-1",
                    "labeled_at": 2.0,
                }
            ),
            encoding="utf-8",
        )
        self.base = [
            "--db",
            str(self.db),
            "--relation-type",
            "phase",
            "--law",
            "cyclic:3",
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def call(self, arguments):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_observe_outcome_health_and_decision_survive_process_boundaries(self):
        code, output, error = self.call(
            [*self.base, "observe", str(self.observation)]
        )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(output)["observation_id"], "obs-1")

        code, output, error = self.call(
            [*self.base, "outcome", str(self.outcome)]
        )
        self.assertEqual((code, error), (0, ""))
        self.assertTrue(json.loads(output)["recorded"])

        code, output, error = self.call(
            ["--db", str(self.db), "health", "--namespace", "run"]
        )
        self.assertEqual((code, error), (0, ""))
        health = json.loads(output)
        overall = health["relations"]["phase"]["overall"]
        self.assertEqual(overall["counts"]["tracked"], 1)
        self.assertEqual(overall["counts"]["labeled"], 1)

        code, output, error = self.call(
            [
                *self.base,
                "decision",
                str(self.query),
                "--focus-node",
                "b",
                "--alternatives",
                "1",
            ]
        )
        self.assertEqual((code, error), (0, ""))
        decision = json.loads(output)
        self.assertEqual(decision["highest_probability_state"], "1")
        self.assertEqual(decision["instructions"], [])

    def test_conflicting_relation_type_is_rejected_without_traceback(self):
        code, output, error = self.call(
            [
                "--db",
                str(self.db),
                "--relation-type",
                "other",
                "--law",
                "cyclic:3",
                "observe",
                str(self.observation),
            ]
        )
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        payload = json.loads(error)
        self.assertEqual(payload["error"]["code"], "input")
        self.assertNotIn("Traceback", error)

    def test_schema_check_and_verify_are_machine_readable(self):
        self.call([*self.base, "observe", str(self.observation)])
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = schema_main(
                ["--db", str(self.db), "check"]
            )
        self.assertEqual((code, stderr.getvalue()), (0, ""))
        self.assertEqual(
            json.loads(stdout.getvalue())["version"],
            CURRENT_SCHEMA_VERSION,
        )

        code, output, error = self.call(
            ["--db", str(self.db), "verify"]
        )
        self.assertEqual((code, error), (0, ""))
        self.assertTrue(json.loads(output)["ok"])

    def test_schema_check_and_verify_fail_on_missing_append_only_trigger(self):
        self.call([*self.base, "observe", str(self.observation)])
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute(
                "DROP TRIGGER monitor_outcomes_no_update"
            )

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = schema_main(["--db", str(self.db), "check"])
        self.assertEqual(code, 3)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue())["error"]["code"], "schema")

        code, output, error = self.call(
            ["--db", str(self.db), "verify"]
        )
        self.assertEqual((code, error), (3, ""))
        payload = json.loads(output)
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["checks"]["schema_compatible"])


if __name__ == "__main__":
    unittest.main()
