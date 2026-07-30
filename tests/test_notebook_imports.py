"""Every notebook's agentlite imports must resolve.

Issue #36: the tutorials import agentlite.logging.multi_agent_log, renamed to
terminal_logger in 96af76c (2024-04-22). It is the first import in the first
cell, so every tutorial failed immediately on a clean checkout.

Only import statements are executed -- not the notebooks, which need an API
key and a network. That is enough to catch a module that does not exist.

Run with:  python -m unittest tests.test_notebook_imports -v
"""

import json
import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK_DIRS = ["tutorials", "example"]
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+agentlite")


def notebooks():
    found = []
    for directory in NOTEBOOK_DIRS:
        found.extend(sorted((REPO_ROOT / directory).glob("*.ipynb")))
    return found


def agentlite_imports(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return [
        line.rstrip("\n")
        for cell in cells
        if cell["cell_type"] == "code"
        for line in cell["source"]
        if IMPORT_RE.match(line)
    ]


class TestNotebookImports(unittest.TestCase):
    def test_there_are_notebooks_to_check(self):
        """Guard against the glob silently matching nothing."""
        self.assertGreaterEqual(len(notebooks()), 8)

    def test_every_agentlite_import_resolves(self):
        failures = []
        for path in notebooks():
            for line in agentlite_imports(path):
                try:
                    exec(line, {})
                except Exception as e:
                    rel = path.relative_to(REPO_ROOT)
                    failures.append(f"{rel}: {line.strip()} -> {type(e).__name__}: {e}")
        self.assertEqual(failures, [], "\n" + "\n".join(failures))

    def test_no_notebook_references_the_renamed_module(self):
        stale = [
            str(path.relative_to(REPO_ROOT))
            for path in notebooks()
            if "multi_agent_log" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(stale, [])

    def test_docs_do_not_reference_the_renamed_module(self):
        rst = REPO_ROOT / "docs" / "source" / "agentlite.logging.rst"
        if not rst.exists():
            self.skipTest("docs not present")
        self.assertNotIn("multi_agent_log", rst.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
