"""Assistant protocol version and its compatibility policy.

v1 is the only version. A mismatch is a hard reject rather than a negotiation:
the platform has no deployed assistant clients to stay compatible with, so a
second version would be dead code guarding an event that cannot happen.

The compatibility policy the version enables is deliberately asymmetric, and
both halves are enforced by the model configs in protocol.py and events.py:

* Reading an event: unknown *fields* are ignored, so a newer agent adding an
  optional field does not break an older client. An unknown event ``type`` is
  rejected, because an unrecognized tag means the stream is not the thing the
  reader believes it is parsing.
* Writing anything, and reading any request: strict. ``extra="forbid"``.

``seq`` is emitted on every event but nothing acts on it yet. There is no
resume in v1 -- a dropped stream ends the run and the client starts a new one.
The field exists so resume can be added later without a version bump.
"""

ASSISTANT_PROTOCOL_VERSION = 1
