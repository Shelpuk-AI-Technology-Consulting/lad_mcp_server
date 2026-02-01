from __future__ import annotations

from dataclasses import dataclass


class TokenBudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenBudget:
    effective_context_length: int
    effective_output_budget: int
    overhead_tokens: int

    @property
    def input_budget_tokens(self) -> int:
        return self.effective_context_length - self.effective_output_budget - self.overhead_tokens

    def validate(self) -> None:
        if self.effective_context_length <= 0:
            raise TokenBudgetError("effective_context_length must be positive")
        if self.effective_output_budget <= 0:
            raise TokenBudgetError("effective_output_budget must be positive")
        if self.overhead_tokens < 0:
            raise TokenBudgetError("overhead_tokens must be non-negative")
        if self.input_budget_tokens <= 0:
            raise TokenBudgetError("Model context too small: effective_context_length <= output_budget + overhead")

