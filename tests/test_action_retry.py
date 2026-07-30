"""Tests for retry/backoff and error handling in external actions.

Relates to issue #19 (DuckDuckGo rate limit ends the run) and issue #29
(unhandled wikipedia.exceptions.PageError ends the run).

Everything here is offline: the wikipedia and duckduckgo_search calls are
replaced with stubs, and the retry decorator takes an injectable sleep so the
tests do not actually wait.

Run with:  python -m unittest tests.test_action_retry -v
"""

import unittest
from unittest import mock

import wikipedia

from agentlite.actions.retry import action_error_message, retry_with_backoff
from example.SearchActions import DuckSearch, WikipediaSearch


class _Boom(Exception):
    pass


class TestRetryWithBackoff(unittest.TestCase):
    def test_returns_immediately_on_success(self):
        calls = []

        @retry_with_backoff(_Boom, sleep=lambda s: None)
        def f():
            calls.append(1)
            return "ok"

        self.assertEqual(f(), "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_then_succeeds(self):
        calls = []

        @retry_with_backoff(_Boom, tries=3, sleep=lambda s: None)
        def f():
            calls.append(1)
            if len(calls) < 3:
                raise _Boom("transient")
            return "ok"

        self.assertEqual(f(), "ok")
        self.assertEqual(len(calls), 3)

    def test_reraises_after_exhausting_tries(self):
        calls = []

        @retry_with_backoff(_Boom, tries=3, sleep=lambda s: None)
        def f():
            calls.append(1)
            raise _Boom("permanent")

        with self.assertRaises(_Boom):
            f()
        self.assertEqual(len(calls), 3)

    def test_does_not_retry_unlisted_exceptions(self):
        calls = []

        @retry_with_backoff(_Boom, tries=3, sleep=lambda s: None)
        def f():
            calls.append(1)
            raise ValueError("different")

        with self.assertRaises(ValueError):
            f()
        self.assertEqual(len(calls), 1, "retried an exception it was not asked to")

    def test_delay_grows_and_is_capped(self):
        waits = []

        @retry_with_backoff(
            _Boom, tries=5, base_delay=1.0, max_delay=4.0, jitter=0, sleep=waits.append
        )
        def f():
            raise _Boom("x")

        with self.assertRaises(_Boom):
            f()
        self.assertEqual(waits, [1.0, 2.0, 4.0, 4.0])

    def test_rejects_nonsense_try_count(self):
        with self.assertRaises(ValueError):
            retry_with_backoff(_Boom, tries=0)

    def test_preserves_function_metadata(self):
        @retry_with_backoff(_Boom)
        def documented():
            """docstring"""

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "docstring")


class TestWikipediaSearchResilience(unittest.TestCase):
    """Issue #29: wikipedia.page() on the top search hit raised PageError."""

    def setUp(self):
        self.action = WikipediaSearch()

    def test_page_error_falls_through_to_next_result(self):
        page = mock.Mock(summary="the real summary")
        with mock.patch.object(wikipedia, "search", return_value=["bad", "good"]), \
             mock.patch.object(
                 wikipedia,
                 "page",
                 side_effect=[wikipedia.exceptions.PageError("bad"), page],
             ):
            self.assertEqual(self.action(query="x"), "the real summary")

    def test_all_results_failing_returns_a_message(self):
        with mock.patch.object(wikipedia, "search", return_value=["a", "b"]), \
             mock.patch.object(
                 wikipedia, "page", side_effect=wikipedia.exceptions.PageError("nope")
             ):
            out = self.action(query="x")
        self.assertIn("none resolved to a page", out)

    def test_disambiguation_returns_the_options(self):
        err = wikipedia.exceptions.DisambiguationError("Mercury", ["planet", "element"])
        with mock.patch.object(wikipedia, "search", return_value=["Mercury"]), \
             mock.patch.object(wikipedia, "page", side_effect=err):
            out = self.action(query="Mercury")
        self.assertIn("ambiguous", out)
        self.assertIn("planet", out)

    def test_no_results_is_unchanged(self):
        with mock.patch.object(wikipedia, "search", return_value=[]):
            self.assertEqual(self.action(query="x"), "No results found.")

    def test_search_failure_returns_message_not_exception(self):
        with mock.patch.object(
            wikipedia,
            "search",
            side_effect=wikipedia.exceptions.WikipediaException("upstream down"),
        ):
            out = self.action(query="x")
        self.assertIn("Wikipedia_Search failed", out)

    def test_happy_path_returns_summary(self):
        page = mock.Mock(summary="a summary")
        with mock.patch.object(wikipedia, "search", return_value=["t"]), \
             mock.patch.object(wikipedia, "page", return_value=page):
            self.assertEqual(self.action(query="x"), "a summary")


class TestDuckSearchResilience(unittest.TestCase):
    """Issue #19: a rate limit on the first call ended the run."""

    def setUp(self):
        self.action = DuckSearch()
        # The decorator resolves time.sleep at call time, so this keeps the
        # backoff from actually delaying the suite.
        patcher = mock.patch("time.sleep")
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_rate_limit_is_retried_then_reported(self):
        from duckduckgo_search.exceptions import RatelimitException

        with mock.patch.object(
            self.action.ddgs, "text", side_effect=RatelimitException("429")
        ) as text:
            out = self.action(query="x")
        self.assertGreater(text.call_count, 1, "gave up without retrying")
        self.assertIn("DuckDuckGo_Search failed", out)

    def test_rate_limit_then_success(self):
        from duckduckgo_search.exceptions import RatelimitException

        results = [{"title": "T", "body": "B"}]
        with mock.patch.object(
            self.action.ddgs, "text", side_effect=[RatelimitException("429"), results]
        ):
            out = self.action(query="x")
        self.assertIn("T", out)
        self.assertIn("B", out)

    def test_uses_the_search_endpoint_not_the_chat_endpoint(self):
        with mock.patch.object(self.action.ddgs, "text", return_value=[]) as text, \
             mock.patch.object(self.action.ddgs, "chat") as chat:
            self.action(query="x")
        text.assert_called_once()
        chat.assert_not_called()


class TestActionErrorMessage(unittest.TestCase):
    def test_names_the_action_the_query_and_the_error(self):
        msg = action_error_message("My_Action", "some query", ValueError("boom"))
        for fragment in ("My_Action", "some query", "ValueError", "boom"):
            self.assertIn(fragment, msg)


if __name__ == "__main__":
    unittest.main()
