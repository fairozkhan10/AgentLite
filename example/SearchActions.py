import os

import wikipedia
import duckduckgo_search
from duckduckgo_search.exceptions import (
    DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)

from agentlite.actions.BaseAction import BaseAction
from agentlite.actions.retry import action_error_message, retry_with_backoff

DDG_RETRY_ON = (RatelimitException, TimeoutException)
WIKI_RETRY_ON = (wikipedia.exceptions.HTTPTimeoutError,)
MAX_RESULTS = 5


class DuckSearch(BaseAction):
    def __init__(self) -> None:
        action_name = "DuckDuckGo_Search"
        action_desc = "Using this action to search online content."
        params_doc = {"query": "the search string. be simple."}
        self.ddgs = duckduckgo_search.DDGS()
        super().__init__(
            action_name=action_name, action_desc=action_desc, params_doc=params_doc,
        )

    @retry_with_backoff(DDG_RETRY_ON)
    def _search(self, query):
        # .text() is the search endpoint. This used to call .chat(), DuckDuckGo's
        # LLM endpoint, which does not perform a search and is rate-limited far
        # more aggressively -- the immediate RatelimitException in issue #19.
        return self.ddgs.text(query, max_results=MAX_RESULTS)

    def __call__(self, query):
        try:
            results = self._search(query)
        # RatelimitException and TimeoutException both subclass this, so this
        # catches a give-up after the retries as well as non-retryable failures.
        except DuckDuckGoSearchException as e:
            return action_error_message(self.action_name, query, e)
        if not results:
            return "No results found."
        return "\n\n".join(
            f"{r.get('title', '')}\n{r.get('body', '')}" for r in results
        )


class WikipediaSearch(BaseAction):
    def __init__(self) -> None:
        action_name = "Wikipedia_Search"
        action_desc = "Using this API to search Wiki content."
        params_doc = {"query": "the search string. be simple."}

        super().__init__(
            action_name=action_name, action_desc=action_desc, params_doc=params_doc,
        )

    @retry_with_backoff(WIKI_RETRY_ON)
    def _search(self, query):
        return wikipedia.search(query)

    @retry_with_backoff(WIKI_RETRY_ON)
    def _page(self, title):
        return wikipedia.page(title)

    def __call__(self, query):
        try:
            search_results = self._search(query)
        except wikipedia.exceptions.WikipediaException as e:
            return action_error_message(self.action_name, query, e)

        if not search_results:
            return "No results found."

        # A search hit is not guaranteed to resolve to a page: the title can be
        # ambiguous or stale, and wikipedia.page() then raises. Previously only
        # the top hit was tried and the exception ended the run (issue #29).
        for title in search_results[:MAX_RESULTS]:
            try:
                return self._page(title).summary
            except wikipedia.exceptions.DisambiguationError as e:
                options = ", ".join(e.options[:MAX_RESULTS])
                return (
                    f"{title!r} is ambiguous. Try one of these more specific "
                    f"queries: {options}."
                )
            except wikipedia.exceptions.PageError:
                continue
            except wikipedia.exceptions.WikipediaException as e:
                return action_error_message(self.action_name, query, e)

        return (
            f"Found {len(search_results)} search results for {query!r} but none "
            f"resolved to a page. Try a different query."
        )
