"""One budget for a whole run, shared by the director and every specialist.

Ceilings that live in a prompt are suggestions. These are counters the loop
checks before it spends, so a director that asks for five consults gets two and
a specialist can never use the call the director needs to answer with.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunBudget:
    max_model_calls: int
    max_consults: int = 0
    # A consult that reviews colleagues' findings. One round: disagreement is
    # useful once and a loop after that.
    max_critique_rounds: int = 1
    model_calls_used: int = 0
    consults_used: int = 0
    critique_rounds_used: int = 0

    def take_director_call(self) -> bool:
        if self.model_calls_used >= self.max_model_calls:
            return False
        self.model_calls_used += 1
        return True

    def take_consult(self) -> bool:
        """Claim one consult and the model call it needs, or neither.

        One call is always held back, so the director can still synthesize an
        answer after the last specialist returns.
        """
        if self.consults_used >= self.max_consults:
            return False
        if self.model_calls_used >= self.max_model_calls - 1:
            return False
        self.consults_used += 1
        self.model_calls_used += 1
        return True

    def take_critique(self) -> bool:
        if self.critique_rounds_used >= self.max_critique_rounds:
            return False
        self.critique_rounds_used += 1
        return True

    def refund_critique(self) -> None:
        self.critique_rounds_used = max(0, self.critique_rounds_used - 1)

    def refund_consult(self) -> None:
        """Give back a consult that never reached the model."""
        self.consults_used = max(0, self.consults_used - 1)
        self.model_calls_used = max(0, self.model_calls_used - 1)
