import importlib.resources
import json
from pathlib import Path
import tomllib
import unittest

import posterior_memory_harness
from posterior_memory_harness import INPUT_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


class PackageResourceTests(unittest.TestCase):
    def test_version_metadata_is_consistent(self):
        metadata = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            posterior_memory_harness.__version__,
            metadata["project"]["version"],
        )

    def test_installed_json_schemas_match_source_contracts(self):
        package_root = importlib.resources.files(
            "posterior_memory_harness"
        )
        for name in (
            "observation.schema.json",
            "query.schema.json",
            "outcome.schema.json",
        ):
            packaged = (
                package_root.joinpath("schemas", name)
                .read_text(encoding="utf-8")
            )
            source = (ROOT / "schemas" / name).read_text(encoding="utf-8")
            self.assertEqual(json.loads(packaged), json.loads(source))
            self.assertEqual(
                json.loads(packaged)["properties"]["schema_version"][
                    "const"
                ],
                INPUT_SCHEMA_VERSION,
            )


if __name__ == "__main__":
    unittest.main()
