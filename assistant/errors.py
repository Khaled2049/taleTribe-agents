"""The closed set of error codes the assistant protocol may put on the wire.

Every member is safe to render to a user as-is. That is the whole point: the
Phase 0 logging audit removed provider bodies, exception messages and prompt
text from logs, and the same rule applies to anything crossing the stream
boundary. Callers pick a code; they never compose the message, because a
composed message is how upstream text leaks back out.
"""

from enum import Enum


class ErrorCode(str, Enum):
    UNSUPPORTED_PROTOCOL_VERSION = "unsupported_protocol_version"
    STORY_ACCESS_DENIED = "story_access_denied"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_ERROR = "provider_error"
    RUN_CANCELLED = "run_cancelled"
    STALE_PROPOSAL = "stale_proposal"
    INTERNAL_ERROR = "internal_error"


_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.UNSUPPORTED_PROTOCOL_VERSION: (
        "This version of the app is out of date. Reload to continue."
    ),
    ErrorCode.STORY_ACCESS_DENIED: "That story is not available.",
    ErrorCode.QUOTA_EXCEEDED: (
        "You have used your AI allowance for today. It resets at midnight UTC."
    ),
    ErrorCode.RATE_LIMITED: "Too many requests. Wait a moment and try again.",
    ErrorCode.PROVIDER_UNAVAILABLE: (
        "The AI service is unreachable right now. Try again shortly."
    ),
    ErrorCode.PROVIDER_ERROR: "The AI service could not complete this request.",
    ErrorCode.RUN_CANCELLED: "Stopped.",
    ErrorCode.STALE_PROPOSAL: (
        "The document changed since this suggestion was made. Ask again for an "
        "up-to-date version."
    ),
    ErrorCode.INTERNAL_ERROR: "Something went wrong. Try again.",
}


def safe_message(code: ErrorCode) -> str:
    """The user-facing text for a code. Total over ErrorCode by construction."""
    return _MESSAGES[code]
