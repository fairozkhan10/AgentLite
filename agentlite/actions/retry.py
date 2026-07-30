"""Retry helper for actions that call flaky external services.

Actions wrap network APIs, and network APIs rate-limit and fail. AgentLite had
no handling for either, so a single transient error propagated out of the
action, out of BaseAgent.forward, and ended the run (issues #19 and #29).

Two pieces:

- :func:`retry_with_backoff`, a decorator that retries on the exceptions you
  name, with exponential backoff and jitter.
- :func:`action_error_message`, for turning a give-up into an observation the
  agent can read and react to, rather than a traceback.
"""

import functools
import random
import time

DEFAULT_TRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0


def retry_with_backoff(
    exceptions,
    tries: int = DEFAULT_TRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = 0.1,
    sleep=None,
):
    """Retry the wrapped callable on `exceptions` with exponential backoff.

    :param exceptions: exception type, or tuple of them, that should be retried
    :type exceptions: type | tuple[type, ...]
    :param tries: total attempts including the first, defaults to 3
    :type tries: int, optional
    :param base_delay: seconds to wait after the first failure, doubling each
        time, defaults to 1.0
    :type base_delay: float, optional
    :param max_delay: upper bound on any single wait, defaults to 30.0
    :type max_delay: float, optional
    :param jitter: fraction of the delay to randomise, to avoid a fleet of
        agents retrying in lockstep, defaults to 0.1
    :type jitter: float, optional
    :param sleep: sleep function, injectable so tests do not have to wait.
        Defaults to looking up time.sleep at call time, so patching
        `time.sleep` works on already-decorated functions.
    :type sleep: callable, optional
    :raises ValueError: if `tries` is less than 1
    :return: a decorator
    :rtype: callable

    The exception from the final attempt is re-raised, so callers that want a
    message instead of an exception should catch it themselves.
    """
    if tries < 1:
        raise ValueError(f"tries must be >= 1, got {tries}")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _sleep = sleep if sleep is not None else time.sleep
            delay = base_delay
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == tries:
                        raise
                    wait = min(delay, max_delay)
                    if jitter:
                        wait += random.uniform(0, wait * jitter)
                    _sleep(wait)
                    delay *= 2

        return wrapper

    return decorator


def action_error_message(action_name: str, query, error: Exception) -> str:
    """Describe a failed action to the agent as a readable observation.

    The agent loop feeds observations back to the model, so a failure that
    explains itself lets the model try a different query or a different action
    on the next step instead of the run ending.
    """
    return (
        f"{action_name} failed for {query!r}: "
        f"{type(error).__name__}: {error}. "
        f"You may retry with a different query, or use another action."
    )
