"""Tests for action parsing and parse-failure handling.

Relates to issue #17, which asks for the agent to recover when the action
parser fails instead of giving up on the step.

No API key or network needed: __action_parser__ and forward() take the raw
string directly, so llm=None is sufficient.

Run with:  python -m unittest tests.test_action_parsing -v
"""

import unittest

from agentlite.agents import BaseAgent, ManagerAgent
from agentlite.agents.agent_utils import (
    ACTION_NOT_FOUND_MESS,
    PARSE_FAILED_MESS,
    parse_action,
)
from agentlite.commons import TaskPackage

UNPARSEABLE = "I think I should look this up first."
WELL_FORMED = 'Finish[{"response": "42"}]'


class TestManagerAgentParser(unittest.TestCase):
    def setUp(self):
        self.manager = ManagerAgent(llm=None, name="M")
        self.task = TaskPackage(instruction="x")

    def test_unparseable_output_does_not_raise(self):
        """Previously raised UnboundLocalError: agent_act was bound only inside
        the match branches, and an unparseable generation matches nothing."""
        act = self.manager.__action_parser__(UNPARSEABLE)
        self.assertTrue(act.parse_failed)

    def test_unmatched_but_parseable_action_does_not_raise(self):
        act = self.manager.__action_parser__('Teleport[{"to": "mars"}]')
        self.assertEqual(act.name, "Teleport")
        self.assertFalse(act.parse_failed)

    def test_forward_returns_corrective_message(self):
        act = self.manager.__action_parser__(UNPARSEABLE)
        self.assertEqual(self.manager.forward(self.task, act), PARSE_FAILED_MESS)


class TestBaseAgentParser(unittest.TestCase):
    def setUp(self):
        self.agent = BaseAgent(name="b", role="r", llm=None)
        self.task = TaskPackage(instruction="x")

    def test_parse_failure_is_recorded(self):
        """PARSE_FLAG was discarded, so a parse failure silently became an
        action named after the whole raw LLM string."""
        self.assertTrue(self.agent.__action_parser__(UNPARSEABLE).parse_failed)

    def test_parse_failure_is_distinct_from_unknown_action(self):
        task = self.task
        unparseable = self.agent.__action_parser__(UNPARSEABLE)
        unknown = self.agent.__action_parser__('Teleport[{"to": "mars"}]')
        self.assertEqual(self.agent.forward(task, unparseable), PARSE_FAILED_MESS)
        self.assertEqual(self.agent.forward(task, unknown), ACTION_NOT_FOUND_MESS)

    def test_well_formed_action_still_executes(self):
        act = self.agent.__action_parser__(WELL_FORMED)
        self.assertEqual(act.name, "Finish")
        self.assertFalse(act.parse_failed)
        self.assertEqual(self.agent.forward(self.task, act), "42")
        self.assertEqual(self.task.completion, "completed")

    def test_invalid_json_arguments_are_a_parse_failure(self):
        """Issue #17's case: multiline content the model failed to JSON-escape."""
        act = self.agent.__action_parser__('Finish[{"response": "line1\nline2"}]')
        self.assertTrue(act.parse_failed)
        self.assertEqual(self.agent.forward(self.task, act), PARSE_FAILED_MESS)


class TestParseActionContract(unittest.TestCase):
    def test_flag_is_false_on_garbage(self):
        self.assertEqual(parse_action(UNPARSEABLE)[2], False)

    def test_flag_is_true_on_well_formed(self):
        name, args, flag = parse_action(WELL_FORMED)
        self.assertEqual((name, args, flag), ("Finish", {"response": "42"}, True))

    def test_action_not_found_message_has_no_stray_quote(self):
        self.assertFalse(ACTION_NOT_FOUND_MESS.startswith('"'))


if __name__ == "__main__":
    unittest.main()
