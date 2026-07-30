"""Tests for non-ASCII handling in prompts and logs.

Covers issue #8 (编码问题), which reports that Chinese text is mangled in the
generated prompt and that logging Chinese can raise.

Run with:  python -m unittest tests.test_encoding -v
"""

import json
import os
import tempfile
import unittest

import agentlite.agents  # noqa: F401  -- importing prompt_utils first is circular

from agentlite.agent_prompts.prompt_utils import action_format
from agentlite.commons import AgentAct
from agentlite.logging.terminal_logger import AgentLogger

CHINESE_TASK = "为小明办理入职手续"
AGENT_NAME = "人事专员"


class TestPromptEncoding(unittest.TestCase):
    """json.dumps escapes non-ASCII by default, so the prompt showed the model
    backslash-u escapes instead of the characters it had just produced."""

    def test_action_params_keep_their_characters(self):
        act = AgentAct(name=AGENT_NAME, params={"Task": CHINESE_TASK})
        formatted = action_format(act)
        self.assertIn(CHINESE_TASK, formatted)

    def test_action_params_contain_no_escape_sequences(self):
        act = AgentAct(name=AGENT_NAME, params={"Task": CHINESE_TASK})
        self.assertNotIn("\\u", action_format(act))

    def test_formatted_action_is_still_valid_json_in_brackets(self):
        act = AgentAct(name=AGENT_NAME, params={"Task": CHINESE_TASK})
        formatted = action_format(act, action_trigger=False)
        payload = formatted[formatted.index("[") + 1 : formatted.rindex("]")]
        self.assertEqual(json.loads(payload), {"Task": CHINESE_TASK})

    def test_ascii_params_are_unchanged(self):
        act = AgentAct(name="Finish", params={"response": "42"})
        self.assertEqual(action_format(act), 'Action:Finish[{"response": "42"}]')


class TestLogFileEncoding(unittest.TestCase):
    """The log file was opened without an explicit encoding, so the platform
    default applied -- cp936 on a Chinese Windows install."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".log")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.path) and os.unlink(self.path))

    def test_chinese_round_trips_through_the_log_file(self):
        logger = AgentLogger(log_file_name=self.path, FLAG_PRINT=False)
        logger.get_obs(CHINESE_TASK)
        with open(self.path, encoding="utf-8") as f:
            self.assertIn(CHINESE_TASK, f.read())

    def test_mixed_scripts_do_not_raise(self):
        logger = AgentLogger(log_file_name=self.path, FLAG_PRINT=False)
        text = "入职 — café — Ωmega — 🙂"
        logger.get_obs(text)
        with open(self.path, encoding="utf-8") as f:
            self.assertIn(text, f.read())

    def test_log_file_is_written_as_utf8_regardless_of_locale(self):
        """Reading the file as UTF-8 must work even if the ambient locale is
        something else, which is what the explicit encoding guarantees."""
        logger = AgentLogger(log_file_name=self.path, FLAG_PRINT=False)
        logger.get_obs(CHINESE_TASK)
        with open(self.path, "rb") as f:
            raw = f.read()
        self.assertIn(CHINESE_TASK.encode("utf-8"), raw)


if __name__ == "__main__":
    unittest.main()
