"""Regression tests for state accidentally shared between instances.

Every test here fails on the parent commit. None of them need an API key or a
network connection -- agents are constructed with llm=None, which is enough to
exercise __init__.

Run with:  python -m unittest tests.test_shared_state -v
"""

import subprocess
import sys
import unittest

from agentlite.actions import BaseAction
from agentlite.agents import BaseAgent, ManagerAgent
from agentlite.commons import TaskPackage


class _NoopAction(BaseAction):
    def __init__(self, name="Noop"):
        super().__init__(action_name=name, action_desc="test action", params_doc={})

    def __call__(self, **kwargs):
        return ""


class TestTaskPackageDefaults(unittest.TestCase):
    def test_task_id_is_unique_per_instance(self):
        """task_id was a class-body default, evaluated once at import time."""
        ids = {TaskPackage(instruction=f"task {i}").task_id for i in range(100)}
        self.assertEqual(len(ids), 100, "task_id is shared between TaskPackages")

    def test_timestamp_is_per_instance_and_a_string(self):
        a = TaskPackage(instruction="one")
        b = TaskPackage(instruction="two")
        self.assertIsInstance(a.timestamp, str)
        self.assertNotEqual(a.timestamp, b.timestamp)

    def test_explicit_task_id_still_honoured(self):
        self.assertEqual(TaskPackage(instruction="x", task_id="fixed").task_id, "fixed")


class TestTaskPackageProvenance(unittest.TestCase):
    """Callers pass task_creator=/task_executor=; the fields are creator/executor.

    Pydantic ignores unknown keyword arguments, so the values were dropped and
    both fields were always "".
    """

    def test_task_creator_alias_is_recorded(self):
        task = TaskPackage(instruction="x", task_creator="User")
        self.assertEqual(task.creator, "User")

    def test_task_executor_alias_is_recorded(self):
        task = TaskPackage(instruction="x", task_executor="Worker")
        self.assertEqual(task.executor, "Worker")

    def test_canonical_field_names_still_work(self):
        task = TaskPackage(instruction="x", creator="A", executor="B")
        self.assertEqual((task.creator, task.executor), ("A", "B"))


class TestBaseAgentActionIsolation(unittest.TestCase):
    def test_caller_action_list_is_not_mutated(self):
        actions = [_NoopAction()]
        BaseAgent(name="a1", role="r", llm=None, actions=actions)
        self.assertEqual(
            [a.action_name for a in actions],
            ["Noop"],
            "constructing an agent appended inner actions to the caller's list",
        )

    def test_reasoning_type_is_respected_for_second_agent(self):
        """The leaked list gave agent two the inner actions of agent one."""
        actions = [_NoopAction()]
        BaseAgent(name="a1", role="r", llm=None, actions=actions, reasoning_type="react")
        a2 = BaseAgent(name="a2", role="r", llm=None, actions=actions, reasoning_type="act")
        self.assertNotIn(
            "Think",
            [a.action_name for a in a2.actions],
            "agent with reasoning_type='act' received ThinkAct",
        )

    def test_default_action_list_is_not_shared(self):
        a1 = BaseAgent(name="a1", role="r", llm=None)
        a2 = BaseAgent(name="a2", role="r", llm=None)
        self.assertIsNot(a1.actions, a2.actions)

    def test_action_order_is_deterministic(self):
        """list(set(...)) ordered by object hash, so prompts varied per run.

        This has to cross process boundaries: the action singletons hash by
        id(), which is stable within one interpreter but not between them.
        """
        script = (
            "from agentlite.agents import BaseAgent;"
            "print(','.join(a.action_name for a in "
            "BaseAgent(name='a', role='r', llm=None).actions))"
        )
        orders = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for _ in range(8)
        }
        self.assertEqual(orders, {"Think,Finish"}, f"unstable action order: {orders}")


class TestManagerAgentTeamIsolation(unittest.TestCase):
    def test_team_is_not_shared_between_managers(self):
        worker = BaseAgent(name="worker", role="r", llm=None)
        m1 = ManagerAgent(llm=None, name="M1")
        m1.add_member(worker)
        m2 = ManagerAgent(llm=None, name="M2")
        self.assertEqual(
            [a.name for a in m2.team], [], "a new manager inherited another's team"
        )

    def test_caller_team_list_is_not_mutated(self):
        team = []
        m = ManagerAgent(llm=None, name="M", TeamAgents=team)
        m.add_member(BaseAgent(name="worker", role="r", llm=None))
        self.assertEqual(team, [], "add_member mutated the caller's list")


if __name__ == "__main__":
    unittest.main()
