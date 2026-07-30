"""Structured, machine-readable agent logging.

AgentLogger already persists to disk, but it writes the same coloured prose it
prints to the terminal:

    Agent wiki_search_agent takes 1-step Action:
    {
        name: Wikipedia_Search
        params: {'query': "Arthur's Magazine"}
    }

That is pleasant to read and unusable as data. Answering "how often does this
agent call each action", "which step did the run fail on", or "how many tokens
of observation did we feed back" means writing a regex against a format that
was never specified, and the `params` field is a Python repr rather than JSON.

JSONAgentLogger implements the same BaseAgentLogger interface and writes one
JSON object per line (JSONL). It is a drop-in alternative:

    from agentlite.logging.json_logger import JSONAgentLogger

    agent = BaseAgent(..., logger=JSONAgentLogger(log_file_name="run.jsonl"))

Every record carries `ts`, `run_id`, `event`, and where known `agent`,
`task_id` and `step_idx`, so a run can be reconstructed, filtered and
aggregated without parsing prose. :func:`read_log` loads one back.

Nesting is tracked: a ManagerAgent delegating to a labor agent produces records
whose `task_id` reflects the task actually executing at that moment, which is
what makes per-agent attribution in a multi-agent run possible.
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

from agentlite.commons import AgentAct, TaskPackage

from .base import BaseAgentLogger


class JSONAgentLogger(BaseAgentLogger):
    """Write the agent event stream as JSON Lines.

    :param log_file_name: file to append to, defaults to "agent.jsonl"
    :type log_file_name: str, optional
    :param FLAG_PRINT: also echo each record to stdout, defaults to False
    :type FLAG_PRINT: bool, optional
    :param OBS_OFFSET: truncate observations longer than this, defaults to 99999
    :type OBS_OFFSET: int, optional
    :param PROMPT_DEBUG_FLAG: record prompts and raw LLM output, defaults to
        False. These are large, so they are opt-in, matching AgentLogger.
    :type PROMPT_DEBUG_FLAG: bool, optional
    :param run_id: identifier shared by every record of this run, defaults to a
        fresh uuid4
    :type run_id: str, optional
    """

    def __init__(
        self,
        log_file_name: str = "agent.jsonl",
        FLAG_PRINT: bool = False,
        OBS_OFFSET: int = 99999,
        PROMPT_DEBUG_FLAG: bool = False,
        run_id: str = None,
    ) -> None:
        super().__init__(log_file_name=log_file_name)
        self.FLAG_PRINT = FLAG_PRINT
        self.OBS_OFFSET = OBS_OFFSET
        self.PROMPT_DEBUG_FLAG = PROMPT_DEBUG_FLAG
        self.run_id = run_id or str(uuid.uuid4())
        # Agents nest: a manager delegates to a labor agent, which executes its
        # own TaskPackage. The stack keeps events attributed to the task that is
        # actually running, since most logger hooks are not passed the task.
        self._task_stack = []
        self._step_idx = None
        # Re-entrant, and held across the whole of each hook: the stack and the
        # step index are read while building a record and mutated when a task
        # starts or finishes, so guarding only the file write would leave the
        # attribution state racy while looking safe.
        self._lock = threading.RLock()

    # -- plumbing ---------------------------------------------------------

    def __save_log__(self, log_str: str):
        """Write a pre-serialised record. Kept for BaseAgentLogger compatibility."""
        with self._lock:
            if self.FLAG_PRINT:
                print(log_str)
            with open(self.log_file_name, "a", encoding="utf-8") as f:
                f.write(log_str + "\n")

    def __emit__(self, event: str, **fields):
        with self._lock:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "event": event,
            }
            if self._task_stack:
                record["task_id"] = self._task_stack[-1]
            if self._step_idx is not None:
                record["step_idx"] = self._step_idx
            # Explicit fields win over the inferred ones.
            record.update({k: v for k, v in fields.items() if v is not None})
            self.__save_log__(json.dumps(record, default=str, ensure_ascii=False))

    # -- BaseAgentLogger interface ---------------------------------------

    def receive_task(self, task: TaskPackage, agent_name: str):
        self.__emit__(
            "receive_task",
            agent=agent_name,
            task_id=task.task_id,
            instruction=task.instruction,
            creator=task.creator or None,
        )

    def execute_task(self, task: TaskPackage = None, agent_name: str = None, **kwargs):
        with self._lock:
            if task is not None:
                self._task_stack.append(task.task_id)
            self._step_idx = None
            self.__emit__("execute_task", agent=agent_name)

    def end_execute(self, task: TaskPackage, agent_name: str = None):
        with self._lock:
            self.__emit__(
                "end_execute",
                agent=agent_name,
                task_id=task.task_id,
                completion=task.completion,
                answer=task.answer,
            )
            self.__unwind__(task.task_id)
            self._step_idx = None

    def __unwind__(self, task_id: str):
        """Drop this task and anything still open beneath it.

        The logger cannot force end_execute to be called: if an action raises,
        the agent loop unwinds without it and that task's entry would otherwise
        stay on the stack forever, mis-attributing every later record. Popping
        from the innermost occurrence of this id -- rather than only when it is
        on top -- discards those orphans when the outer task does finish.
        """
        if task_id in self._task_stack:
            innermost = len(self._task_stack) - 1 - self._task_stack[::-1].index(task_id)
            del self._task_stack[innermost:]

    def reset(self):
        """Forget any in-progress task state. For reusing one logger across runs."""
        with self._lock:
            self._task_stack.clear()
            self._step_idx = None

    def take_action(self, action: AgentAct, agent_name: str, step_idx: int):
        with self._lock:
            self._step_idx = step_idx
        self.__emit__(
            "take_action",
            agent=agent_name,
            action=action.name,
            params=action.params,
            parse_failed=getattr(action, "parse_failed", None) or None,
        )

    def add_st_memory(self, agent_name: str):
        # AgentLogger exposes this and nothing in the framework calls it, but it
        # is public API -- without it, swapping this logger in raises
        # AttributeError for anyone who does.
        self.__emit__("add_st_memory", agent=agent_name)

    def get_obs(self, obs: str):
        truncated = len(obs) > self.OBS_OFFSET
        self.__emit__(
            "observation",
            observation=obs[: self.OBS_OFFSET] if truncated else obs,
            truncated=truncated or None,
            length=len(obs),
        )

    def get_prompt(self, prompt):
        if self.PROMPT_DEBUG_FLAG:
            self.__emit__("prompt", prompt=prompt)

    def get_llm_output(self, output: str):
        if self.PROMPT_DEBUG_FLAG:
            self.__emit__("llm_output", output=output)


def read_log(log_file_name: str) -> list:
    """Load a JSONL agent log back into a list of dicts.

    Blank lines are skipped. Malformed lines raise, rather than being dropped
    silently -- a log that cannot be parsed is worth knowing about.

    :param log_file_name: path written by JSONAgentLogger
    :type log_file_name: str
    :return: the records, in file order
    :rtype: list[dict]
    """
    records = []
    with open(log_file_name, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{log_file_name}:{lineno} is not valid JSON") from e
    return records
