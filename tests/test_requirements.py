"""Tests for the declared dependency set.

These assert the shape of requirements.txt and what setup.py makes of it. They
do not install anything and do not touch the network.

Run with:  python -m unittest tests.test_requirements -v
"""

import contextlib
import os
import pathlib
import unittest

from packaging.requirements import InvalidRequirement, Requirement

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def parse_requirements():
    """Call setup.py's own get_requires().

    Executed rather than reimplemented, so this tests the shipped parser and
    not a copy of it. setup() is stubbed out because setup.py calls it at
    module scope, and get_requires() opens requirements.txt by relative path.
    """
    import sys
    import types

    stub = types.ModuleType("setuptools")
    stub.setup = lambda *a, **kw: None
    stub.find_packages = lambda *a, **kw: []

    source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    namespace = {"__name__": "setup_under_test", "__file__": str(REPO_ROOT / "setup.py")}
    previous = sys.modules.get("setuptools")
    sys.modules["setuptools"] = stub
    cwd = os.getcwd()
    try:
        os.chdir(REPO_ROOT)
        exec(compile(source, "setup.py", "exec"), namespace)
        return namespace["get_requires"]()
    finally:
        os.chdir(cwd)
        if previous is None:
            del sys.modules["setuptools"]
        else:
            sys.modules["setuptools"] = previous


class TestRequirementsAreWellFormed(unittest.TestCase):
    def test_every_entry_is_a_valid_requirement(self):
        """Blank lines used to survive the parser as "" entries, which are not
        valid requirement strings."""
        for entry in parse_requirements():
            try:
                Requirement(entry)
            except InvalidRequirement as e:
                self.fail(f"{entry!r} is not a valid requirement: {e}")

    def test_no_empty_entries(self):
        self.assertNotIn("", parse_requirements())

    def test_comment_lines_are_excluded(self):
        self.assertTrue(all(not e.startswith("#") for e in parse_requirements()))


class TestPinsThatCleanInstallsNeed(unittest.TestCase):
    def requirement(self, name):
        for entry in parse_requirements():
            try:
                req = Requirement(entry)
            except InvalidRequirement:
                continue  # reported by TestRequirementsAreWellFormed
            if req.name.lower().replace("_", "-") == name:
                return req
        self.fail(f"{name} is not declared in requirements.txt")

    def test_httpx_is_bounded_below_0_28(self):
        """openai==1.10 passes proxies= to httpx.Client, removed in httpx 0.28."""
        self.assertTrue(
            self.requirement("httpx").specifier.contains("0.27.2"),
            "httpx 0.27.2 should satisfy the declared bound",
        )
        self.assertFalse(
            self.requirement("httpx").specifier.contains("0.28.0"),
            "httpx 0.28 breaks every LLM backend and must be excluded",
        )

    def test_pyreqwest_impersonate_is_pinned_to_a_wheel_version(self):
        """0.5.5 has no macOS-arm64 wheel and needs a Rust toolchain."""
        spec = self.requirement("pyreqwest-impersonate").specifier
        self.assertTrue(spec.contains("0.4.7"))
        self.assertFalse(spec.contains("0.5.5"))

    def test_openai_and_httpx_are_mutually_satisfiable(self):
        openai = self.requirement("openai")
        self.assertTrue(openai.specifier.contains("1.10.0"))


if __name__ == "__main__":
    unittest.main()
