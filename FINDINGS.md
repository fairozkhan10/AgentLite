# AgentLite: defects found and fixed

A review of [SalesforceAIResearch/AgentLite](https://github.com/SalesforceAIResearch/AgentLite)
at `b173239`, carried out in July 2026.

Every defect below was **confirmed by executing code**, not by reading it. All
reproductions run offline — no API key, no network — which is why they are
included as tests rather than described in prose.

The work is split across eight branches, one per proposed change, each based on
`b173239` and independent of the others.

---

## Summary

| Branch | What it fixes | Upstream issue | Tests |
|---|---|---|---|
| `fix/install-deps` | `pip install -e .` fails; every LLM backend fails to construct | (#31?) | 6 |
| `fix/taskpackage-mutable-defaults` | Five instances of state shared between objects that should be independent | — | 12 |
| `fix/manager-agent-parse-crash` | `UnboundLocalError` on unparseable model output; parse failures indistinguishable from unknown actions | [#17](https://github.com/SalesforceAIResearch/AgentLite/issues/17) | 10 |
| `fix/missing-multi-agent-log` | Every tutorial fails on its first cell | [#36](https://github.com/SalesforceAIResearch/AgentLite/issues/36) | 4 |
| `feat/action-retry-backoff` | A single transient network error ends the run | [#29](https://github.com/SalesforceAIResearch/AgentLite/issues/29), [#19](https://github.com/SalesforceAIResearch/AgentLite/issues/19) | 17 |
| `feat/structured-json-logging` | Logs are prose; agent runs cannot be analysed as data | — | 25 |
| `fix/chinese-encoding` | Non-ASCII is escaped back into the prompt | [#8](https://github.com/SalesforceAIResearch/AgentLite/issues/8) | 7 |
| `ci/github-actions` | No CI exists | — | — |

**81 new tests**, all offline, all under stdlib `unittest` to match the existing
tests and add no test dependency.

Verified on GitHub Actions across Python 3.10 / 3.11 / 3.12, with every branch
checked out and run in turn:
[run #30611975286](https://github.com/fairozkhan10/AgentLite/actions/runs/30611975286) — 10/10 jobs green.

The branches are independent and apply in any order. Merging all eight in
sequence produces no conflicts, and the combined suite — 82 tests — passes with
every change stacked together.

---

## 1. The documented Quick Start does not install

`requirements.txt` pins `langchain==0.1.3` / `openai==1.10` from early 2024 but
leaves their transitive dependencies unbounded. Two of those have since moved.

**`pip install -e .` fails.** `duckduckgo-search==6.1.0` requires
`pyreqwest_impersonate>=0.4.7`; pip resolves to 0.5.5, which ships no
macOS-arm64 wheel and needs a Rust and cmake toolchain to build from source.

**Every LLM backend fails to construct.** `openai==1.10` passes `proxies=` to
`httpx.Client`, an argument httpx removed in 0.28:

```
ValidationError: Client.__init__() got an unexpected keyword argument 'proxies'
```

Fix: bound both. Also filters blank lines out of `setup.py`'s requirements
parser, which passed them into `install_requires` as `""` entries — not valid
requirement strings, tolerated by current setuptools only by luck.

---

## 2. State shared between objects that should be independent

Five defects, one root cause: mutable or once-evaluated values at class
definition scope.

**Every `TaskPackage` in a process has the same `task_id`.**
`commons/TaskPackage.py:15` reads `task_id: str = str(uuid.uuid4())` — a
class-body default, evaluated once at import.

```python
>>> TaskPackage(instruction="one").task_id == TaskPackage(instruction="two").task_id
True
```

This is visible in [issue #29](https://github.com/SalesforceAIResearch/AgentLite/issues/29)'s
own pasted log, where Task ID `03244001-…` appears under two different
questions. It also corrupts `DictAgentSTMemory`, which is keyed on task id.
`timestamp` has the same defect, and additionally held a `float` in a field
annotated `str` — pydantic does not validate defaults.

**Task provenance was silently discarded.** The model declares `creator` and
`executor`; every caller in the tree passes `task_creator=` / `task_executor=`
(`ManagerAgent.py:185`, `example/SearchManager.py:76`, the README, the
multi-agent tutorial). Pydantic ignores unknown keyword arguments, so both
fields were always `""`. Fixed with validation aliases, so existing user code
starts working rather than breaking.

**`BaseAgent(actions=[...])` mutates the caller's list.**
`agents/BaseAgent.py:49` is a shared mutable default and inner actions are
appended in place. Two agents built from one list:

```
before:       ['Noop']
after agent1: ['Noop', 'Think', 'Finish']
after agent2: ['Noop', 'Think', 'Finish', 'Finish']
```

The second agent receives `ThinkAct` despite `reasoning_type="act"`.

**`ManagerAgent` has the same defect.** Its `TeamAgents` parameter defaults to
`[]` (`ManagerAgent.py:26`) and is assigned straight to `self.team`, which
`add_member()` appends to — so every manager constructed without an explicit
team shares one list object:

```
>>> m1, m2 = ManagerAgent(...), ManagerAgent(...)
>>> m1.add_member(a1)
>>> [x.name for x in m2.team]
['a1']
>>> m1.team is m2.team
True
```

**Action order was nondeterministic across processes.**
`list(set(self.actions))` de-duplicates by object hash, i.e. by `id()`. Running
the same constructor in five subprocesses returns both orderings, so the prompt
built from that list differed run to run. Now `dict.fromkeys`, which
de-duplicates identically but preserves insertion order.

---

## 3. Unparseable model output crashes the manager — issue #17

`ManagerAgent.__action_parser__` bound `agent_act` only inside its match
branches, so any output matching neither a labor agent nor an action fell
through to `return agent_act` with nothing bound:

```
UnboundLocalError: cannot access local variable 'agent_act'
```

Every unparseable generation is such a miss — and unparseable generations are
exactly what issue #17 describes: a model emitting multiline content that is not
valid JSON inside `Finish[{...}]`.

`BaseAgent` had the mirror-image problem. `parse_action()` returns a
`PARSE_FLAG`; `BaseAgent.__action_parser__` discarded it. Since `parse_action()`
returns the *raw input string* as the action name on failure, a malformed step
silently became an action named after the entire model output, reported as
"wrong action to call" — indistinguishable from a hallucinated action name, and
telling the model nothing about what was actually wrong.

`AgentAct` now carries `parse_failed`, and both agents return a message
restating the required format. That observation enters the action chain, so the
model can re-emit the step on the next iteration — the recovery path #17 asks
for. Unknown-but-parseable action names still report separately.

---

## 4. Every tutorial fails on its first cell — issue #36

```python
from agentlite.logging.multi_agent_log import AgentLogger
ModuleNotFoundError: No module named 'agentlite.logging.multi_agent_log'
```

`multi_agent_log.py` was renamed to `terminal_logger.py` in `96af76c`
(2024-04-22, `R098`). That commit updated the library and the benchmarks but not
the notebooks or the docs, which have imported a module that has not existed for
two years. It is the first import in the first cell of all eight tutorials and
of `example/SearchAgent.ipynb`.

Verified by executing the `agentlite` import lines of each notebook: **9 of 9
fail before, 0 of 9 after.** `docs/source/agentlite.logging.rst` pointed its
`automodule` directive at the same dead module, so Sphinx could never have
documented it.

---

## 5. One transient network error ends the run — issues #29 and #19

Neither `SearchActions.py` contained a single `try`/`except`. Any failure
propagated out of the action, out of `BaseAgent.forward`, and terminated the
run.

**Issue #29** — `evaluate_hotpot_qa.py` dies partway through the benchmark. The
cause is `wikipedia.page(search_results[0])`: a search hit is not guaranteed to
resolve to a page, and only the top hit was ever tried.

```
PageError: Page id "bad" does not match any pages. Try another id!
```

The action now walks the results and returns the first that resolves, so one
dead title no longer ends a 100-question benchmark. `DisambiguationError`
returns the candidate titles, which is more useful to the model than a failure.

**Issue #19** — DuckDuckGo rate-limits on the very first call. Two causes. The
action called `self.ddgs.chat(query)` — DuckDuckGo's **LLM endpoint, not its
search endpoint**. It does not perform a search and is rate-limited far more
aggressively. That is now `.text(query, max_results=5)`, which is what an action
named `DuckDuckGo_Search` should have been calling. Rate limits are also now
retried with exponential backoff instead of being fatal.

Adds `agentlite/actions/retry.py` — a `retry_with_backoff` decorator and a
helper that renders a give-up as an observation the agent can act on, both
exported for use by user-defined actions.

---

## 6. Non-ASCII is escaped back into the prompt — issue #8

The reporter pasted this from their run:

```
Action:人事专员[{"Task": "为小明办理入职手续"}]
```

The agent name survives; the parameters do not. `action_format()` builds that
string with `json.dumps(act.params)`, which escapes non-ASCII by default. The
string is not merely displayed — `action_chain_format()` feeds it into the next
prompt, so the model is shown a run of escape sequences in place of the
characters it produced one step earlier. It costs several times the tokens, and
it is exactly the "may affect model performance" the reporter suspected.
Reproduced verbatim.

Separately, `AgentLogger` opened its log file with no encoding, so the platform
default applied — `cp936` on the Chinese Windows install this was reported from,
which raises `UnicodeEncodeError` outside its codepage. Now pinned to UTF-8.

---

## 7. No CI

There is no `.github/` directory, no linter, no formatter config, and
`pyproject.toml` does not exist. `tests/` holds two files totalling about thirty
lines, one of which requires a paid API key and calls the retired `gpt-4-32k`.
Nothing verifies a change automatically.

`ci/github-actions` adds a workflow that runs the offline suite on every push
across Python 3.10–3.12, plus a manual job that checks out each branch and runs
its tests, so the whole set is verifiable from one run.

While adding it: `.gitignore` line 7 was `*test*`, which matches **any** path
containing "test" — the `tests/` package and `.github/workflows/tests.yml`
included. No new test file could be committed without `git add -f`. That is a
plausible explanation for why the test suite never grew.

---

## Known issues not addressed

- **Circular import.** Importing `agentlite.agent_prompts.prompt_utils` before
  `agentlite.agents` raises `ImportError`. Real and reproducible; worked around
  in one test file rather than fixed, as it touches package layout.
- **Issue #8, second half.** The reporter also notes the philosophers example
  behaving differently once its prompts are translated. That is a
  prompt-content question, not an encoding one, and needs a live model.
- **`tests/test_llm.py`** targets `gpt-4-32k`, retired. Choosing a replacement
  model is a maintainer decision, so it is excluded from CI rather than changed.
- **Issue #25** asks for `WikiSearchAgent` to be the default in
  `example/SearchAgent.py`. It already is — `SearchAgent.py:107`, with
  `DuckSearchAgent` commented out on the next line. The remaining DuckDuckGo
  exposure is `example/SearchManager.py:41`, whose failure mode is addressed by
  `feat/action-retry-backoff`. No change proposed.
- **Benchmarks** were not run: they require dataset downloads and paid API
  calls.

---

## Limits of this review

Stated plainly, because the above is not a clean bill of health:

- **Nothing was verified against a live model or a live endpoint.** Every test
  here is offline by design, which is what makes them runnable in CI, but it
  means the network-facing changes are the least proven: the DuckDuckGo and
  Wikipedia paths in `feat/action-retry-backoff` are exercised against fakes,
  not against the real services.
- **The `.chat()` → `.text()` switch is the most opinionated change in the
  set.** The evidence that `.chat()` is wrong is strong — it is DuckDuckGo's LLM
  endpoint, in an action named `DuckDuckGo_Search` — but it changes what the
  example returns, and a maintainer may have intended it.
- The review followed the open issues and the modules they touch. Findings 2 and
  the ordering defect were found independently; the rest started from a report.
  This was not a systematic pass over the whole library.
- `feat/structured-json-logging` adds a capability rather than fixing a defect.
  It is included because agent traces being unanalysable as data is a real
  limitation, but it is the one branch that is not a bug fix.

---

## Reproducing this

```bash
git clone https://github.com/fairozkhan10/AgentLite.git
cd AgentLite
python -m venv .venv && . .venv/bin/activate
pip install "httpx<0.28" "pyreqwest-impersonate==0.4.7"
pip install -e .

git checkout fix/taskpackage-mutable-defaults
python -m unittest tests.test_shared_state -v
```

Where a suite can run against `main`, it fails there and passes on the branch:
9 of 12 for `fix/taskpackage-mutable-defaults`, 2 of 7 for `fix/chinese-encoding`
(its three log-file tests pass on macOS and Linux, where the locale default is
already UTF-8 — that fix is hardening for the Windows install the issue came
from).

Three suites cannot run against `main` at all, because they import symbols the
branch introduces — `fix/manager-agent-parse-crash`,
`feat/action-retry-backoff` and `feat/structured-json-logging`. For those, the
original failures are reproduced by the standalone snippets in the commit
messages.
