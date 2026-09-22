# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Agentic rollout package."""

from typing import Any

from relax.utils.log_style import format_badge


# Stable Ray Serve application name. Changing it breaks Runtime handle lookup
# and must be deployed atomically with every borrower.
AGENTIC_CHAT_API_SERVICE_NAME = "agentic_chat_api"
# Stable OpenAI-compatible BaseURL prefix used by external agent processes.
# Changing it requires updating every launched agent configuration together.
AGENTIC_CHAT_API_ROUTE_PREFIX = "/agentic_api"

_ANSI_RESET = "\033[0m"
_ANSI_EVENT = "\033[1;38;5;213m"


def format_agentic_event(component: str, event: str, **fields: Any) -> str:
    """Format one compact, colored Agentic lifecycle event."""

    tokens = [format_badge(f"AGENTIC {component}"), f"{_ANSI_EVENT}event={event}{_ANSI_RESET}"]
    tokens.extend(f"{key}={value}" for key, value in fields.items() if value is not None)
    return " ".join(tokens)


def format_agentic_status(message: str) -> str:
    return f"{format_badge('AGENTIC')} {_ANSI_EVENT}{message}{_ANSI_RESET}"
