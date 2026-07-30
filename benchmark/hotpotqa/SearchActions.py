import os
import wikipedia

from agentlite.actions.BaseAction import BaseAction
from agentlite.actions.retry import action_error_message, retry_with_backoff

WIKI_RETRY_ON = (wikipedia.exceptions.HTTPTimeoutError,)
MAX_RESULTS = 5


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
        # the top hit was tried, and the exception ended the benchmark run --
        # this is the crash in issue #29.
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
