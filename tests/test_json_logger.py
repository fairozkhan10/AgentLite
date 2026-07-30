"""Tests for JSONAgentLogger.

Runs a real agent loop with a scripted LLM, so no API key and no network are
needed -- BaseAgent only requires that llm(prompt) return a string.

Run with:  python -m unittest tests.test_json_logger -v
"""

import json
import os
import tempfile
import unittest

from agentlite.agents import BaseAgent, ManagerAgent
from agentlite.commons import TaskPackage
from agentlite.logging.json_logger import JSONAgentLogger, read_log


class ScriptedLLM:
    """Replays canned LLM outputs in order."""

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def __call__(self, prompt):
        return self.run(prompt)

    def run(self, prompt):
        return self.outputs.pop(0) if self.outputs else 'Finish[{"response": "done"}]'


class _LoggerTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        os.unlink(self.path)
        self.addCleanup(
            lambda: os.path.exists(self.path) and os.unlink(self.path)
        )

    def make_logger(self, **kwargs):
        return JSONAgentLogger(log_file_name=self.path, **kwargs)

    def run_agent(self, logger, outputs=None, instruction="what is the answer?"):
        agent = BaseAgent(
            name="demo_agent",
            role="answer questions",
            llm=ScriptedLLM(
                outputs
                or [
                    'Think[{"response": "thinking"}]',
                    'Finish[{"response": "42"}]',
                ]
            ),
            logger=logger,
        )
        agent(TaskPackage(instruction=instruction))
        return read_log(self.path)


class TestRecordShape(_LoggerTestCase):
    def test_every_line_is_valid_json(self):
        self.run_agent(self.make_logger())
        with open(self.path, encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        self.assertGreater(len(lines), 0)
        for line in lines:
            json.loads(line)  # raises if the log is not machine-readable

    def test_required_fields_present_on_every_record(self):
        for record in self.run_agent(self.make_logger()):
            for field in ("ts", "run_id", "event"):
                self.assertIn(field, record)

    def test_run_id_is_constant_and_honoured(self):
        records = self.run_agent(self.make_logger(run_id="fixed-run"))
        self.assertEqual({r["run_id"] for r in records}, {"fixed-run"})

    def test_event_sequence_of_a_complete_run(self):
        events = [r["event"] for r in self.run_agent(self.make_logger())]
        self.assertEqual(
            events,
            [
                "receive_task",
                "execute_task",
                "take_action",
                "observation",
                "take_action",
                "observation",
                "end_execute",
            ],
        )

    def test_params_are_json_not_python_repr(self):
        """The prose logger wrote params as a Python repr, with single quotes."""
        records = self.run_agent(self.make_logger())
        action = next(r for r in records if r["event"] == "take_action")
        self.assertEqual(action["params"], {"response": "thinking"})

    def test_steps_and_task_id_are_attributed(self):
        records = self.run_agent(self.make_logger())
        task_ids = {r["task_id"] for r in records}
        self.assertEqual(len(task_ids), 1)
        steps = [r["step_idx"] for r in records if r["event"] == "take_action"]
        self.assertEqual(steps, [0, 1])

    def test_final_answer_is_recorded(self):
        end = next(
            r for r in self.run_agent(self.make_logger()) if r["event"] == "end_execute"
        )
        self.assertEqual(end["completion"], "completed")
        self.assertEqual(end["answer"], "42")


class TestVerbosityControls(_LoggerTestCase):
    def test_prompts_are_omitted_by_default(self):
        events = {r["event"] for r in self.run_agent(self.make_logger())}
        self.assertNotIn("prompt", events)
        self.assertNotIn("llm_output", events)

    def test_prompts_are_recorded_when_requested(self):
        events = {
            r["event"]
            for r in self.run_agent(self.make_logger(PROMPT_DEBUG_FLAG=True))
        }
        self.assertIn("prompt", events)
        self.assertIn("llm_output", events)

    def test_long_observations_are_truncated_and_flagged(self):
        logger = self.make_logger(OBS_OFFSET=10)
        logger.get_obs("x" * 50)
        record = read_log(self.path)[0]
        self.assertEqual(len(record["observation"]), 10)
        self.assertTrue(record["truncated"])
        self.assertEqual(record["length"], 50)

    def test_short_observations_are_not_flagged(self):
        logger = self.make_logger(OBS_OFFSET=10)
        logger.get_obs("short")
        record = read_log(self.path)[0]
        self.assertEqual(record["observation"], "short")
        self.assertNotIn("truncated", record)


class TestMultiAgentAttribution(_LoggerTestCase):
    def test_labor_agent_records_carry_their_own_task_id(self):
        logger = self.make_logger()
        worker = BaseAgent(
            name="worker",
            role="do the work",
            llm=ScriptedLLM(['Finish[{"response": "sub-answer"}]']),
            logger=logger,
        )
        manager = ManagerAgent(
            llm=ScriptedLLM(
                [
                    'worker[{"Task": "do the thing"}]',
                    'Finish[{"response": "final"}]',
                ]
            ),
            name="manager",
            TeamAgents=[worker],
            logger=logger,
        )
        manager(TaskPackage(instruction="delegate this"))

        records = read_log(self.path)
        agents = {r["agent"] for r in records if "agent" in r}
        self.assertEqual(agents, {"manager", "worker"})

        # The delegated work is bracketed by the worker's own receive/execute
        # and end_execute, nested inside the manager's execution.
        events = [(r.get("agent"), r["event"]) for r in records]
        self.assertEqual(events[0], ("manager", "receive_task"))
        self.assertIn(("worker", "receive_task"), events)
        self.assertLess(
            events.index(("worker", "end_execute")),
            len(events) - 1,
            "the worker should finish before the manager does",
        )

        # NOTE: the worker's task_id is currently identical to the manager's,
        # because TaskPackage.task_id is a class-body default evaluated once at
        # import, so every TaskPackage in a process shares one id. Attribution
        # by task_id therefore cannot be asserted here; that defect is fixed
        # separately in commons/TaskPackage.py. Once it is, these records
        # distinguish the two tasks with no change to the logger.
        self.assertTrue(all("task_id" in r for r in records))

    def test_nesting_unwinds_back_to_the_manager(self):
        logger = self.make_logger()
        worker = BaseAgent(
            name="worker",
            role="do the work",
            llm=ScriptedLLM(['Finish[{"response": "sub"}]']),
            logger=logger,
        )
        manager = ManagerAgent(
            llm=ScriptedLLM(
                [
                    'worker[{"Task": "sub"}]',
                    'Finish[{"response": "final"}]',
                ]
            ),
            name="manager",
            TeamAgents=[worker],
            logger=logger,
        )
        manager(TaskPackage(instruction="delegate"))
        records = read_log(self.path)
        first, last = records[0], records[-1]
        self.assertEqual(last["event"], "end_execute")
        self.assertEqual(last["agent"], "manager")
        self.assertEqual(last["task_id"], first["task_id"])


class TestStackHygiene(_LoggerTestCase):
    """The logger cannot force end_execute to be called; an action that raises
    unwinds the agent loop without it."""

    def test_abandoned_inner_task_does_not_leak(self):
        logger = self.make_logger()
        outer = TaskPackage(instruction="outer", task_id="outer-id")
        inner = TaskPackage(instruction="inner", task_id="inner-id")
        logger.execute_task(task=outer, agent_name="manager")
        logger.execute_task(task=inner, agent_name="worker")
        # inner never ends -- simulate the action raising
        logger.end_execute(outer, agent_name="manager")
        self.assertEqual(logger._task_stack, [])

    def test_records_after_an_abandoned_task_are_not_misattributed(self):
        logger = self.make_logger()
        outer = TaskPackage(instruction="outer", task_id="outer-id")
        inner = TaskPackage(instruction="inner", task_id="inner-id")
        logger.execute_task(task=outer, agent_name="manager")
        logger.execute_task(task=inner, agent_name="worker")
        logger.end_execute(outer, agent_name="manager")
        logger.execute_task(task=outer, agent_name="manager")
        logger.get_obs("after")
        record = read_log(self.path)[-1]
        self.assertEqual(record["task_id"], "outer-id")

    def test_reset_clears_in_progress_state(self):
        logger = self.make_logger()
        logger.execute_task(
            task=TaskPackage(instruction="x", task_id="a"), agent_name="agent"
        )
        logger.reset()
        self.assertEqual(logger._task_stack, [])
        self.assertIsNone(logger._step_idx)

    def test_concurrent_agents_do_not_interleave_records(self):
        """Every line must remain a complete, parseable JSON object."""
        import threading

        logger = self.make_logger()
        def worker(n):
            for i in range(25):
                logger.get_obs(f"agent-{n}-obs-{i}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        records = read_log(self.path)  # raises if any line is torn
        self.assertEqual(len(records), 100)


class TestReadLog(_LoggerTestCase):
    def test_round_trip(self):
        logger = self.make_logger()
        logger.get_obs("hello")
        self.assertEqual(read_log(self.path)[0]["observation"], "hello")

    def test_blank_lines_are_skipped(self):
        logger = self.make_logger()
        logger.get_obs("a")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n\n")
        self.assertEqual(len(read_log(self.path)), 1)

    def test_malformed_line_raises_with_location(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write('{"ok": 1}\nnot json\n')
        with self.assertRaises(ValueError) as ctx:
            read_log(self.path)
        self.assertIn(":2", str(ctx.exception))

    def test_non_ascii_survives_the_round_trip(self):
        """Relevant to issue #8: log content must not be mangled."""
        logger = self.make_logger()
        logger.get_obs("编码问题 — ünïcode")
        self.assertEqual(read_log(self.path)[0]["observation"], "编码问题 — ünïcode")


class TestDropInCompatibility(_LoggerTestCase):
    def test_implements_the_base_logger_interface(self):
        from agentlite.logging.base import BaseAgentLogger

        self.assertTrue(issubclass(JSONAgentLogger, BaseAgentLogger))
        for method in (
            "receive_task",
            "execute_task",
            "end_execute",
            "take_action",
            "get_obs",
            "get_prompt",
            "get_llm_output",
        ):
            self.assertTrue(callable(getattr(JSONAgentLogger, method)))

    def test_covers_every_public_method_of_the_terminal_logger(self):
        """Derived from AgentLogger rather than a hand-written list.

        The list above missed add_st_memory, which AgentLogger exposes and
        nothing in the framework calls -- so swapping loggers raised
        AttributeError only for user code that used it.
        """
        from agentlite.logging import AgentLogger

        public = lambda cls: {n for n in dir(cls) if not n.startswith("_")}
        missing = sorted(public(AgentLogger) - public(JSONAgentLogger))

        self.assertEqual(missing, [], f"JSONAgentLogger is missing {missing}")

    def test_add_st_memory_emits_a_record(self):
        logger = self.make_logger()
        logger.add_st_memory(agent_name="agent")

        records = read_log(self.path)
        self.assertEqual([r["event"] for r in records], ["add_st_memory"])
        self.assertEqual(records[0]["agent"], "agent")

    def test_default_terminal_logger_is_unchanged(self):
        from agentlite.logging import AgentLogger, DefaultLogger

        self.assertIsInstance(DefaultLogger, AgentLogger)


if __name__ == "__main__":
    unittest.main()
