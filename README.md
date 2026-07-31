# AgentLite — review notes

This is a personal fork of
[SalesforceAIResearch/AgentLite](https://github.com/SalesforceAIResearch/AgentLite).
It is not an official Salesforce repository and is not affiliated with Salesforce.
The project's own README is on [`main`](../../blob/main/README.md), unchanged.

I read through the project at `b173239` in July 2026, reproduced the defects
behind several of its open issues, and fixed each on its own branch.

**→ [FINDINGS.md](FINDINGS.md) — what I found, how each defect reproduces, and
what each fix does.**

| Branch | Fixes | Upstream issue |
|---|---|---|
| [`fix/install-deps`](../../tree/fix/install-deps) | `pip install -e .` fails; every LLM backend fails to construct | — |
| [`fix/taskpackage-mutable-defaults`](../../tree/fix/taskpackage-mutable-defaults) | Five instances of state shared between objects that should be independent | — |
| [`fix/manager-agent-parse-crash`](../../tree/fix/manager-agent-parse-crash) | `UnboundLocalError` on unparseable model output | [#17](https://github.com/SalesforceAIResearch/AgentLite/issues/17) |
| [`fix/missing-multi-agent-log`](../../tree/fix/missing-multi-agent-log) | Every tutorial fails on its first cell | [#36](https://github.com/SalesforceAIResearch/AgentLite/issues/36) |
| [`feat/action-retry-backoff`](../../tree/feat/action-retry-backoff) | A single transient network error ends the run | [#29](https://github.com/SalesforceAIResearch/AgentLite/issues/29), [#19](https://github.com/SalesforceAIResearch/AgentLite/issues/19) |
| [`feat/structured-json-logging`](../../tree/feat/structured-json-logging) | Logs are prose; agent runs cannot be analysed as data | — |
| [`fix/chinese-encoding`](../../tree/fix/chinese-encoding) | Non-ASCII is escaped back into the prompt | [#8](https://github.com/SalesforceAIResearch/AgentLite/issues/8) |
| [`ci/github-actions`](../../tree/ci/github-actions) | No CI exists | — |

79 new tests, all offline — no API key, no network, no model calls. Verified on
GitHub Actions across Python 3.10, 3.11 and 3.12, with every branch checked out
and run in turn: [run #30580156054](../../actions/runs/30580156054), 10/10 jobs
green.

The branches are independent and apply in any order. Merging all of them in
sequence produces no conflicts, and the combined suite — 80 tests — passes with
every change stacked together.

No pull requests have been opened upstream and nothing has been posted to the
project's issue tracker. `main` on this fork is identical to upstream.
