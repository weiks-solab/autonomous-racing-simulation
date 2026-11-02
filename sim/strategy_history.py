"""記錄每台車的策略變化，讓 main.py 專注組裝與執行。"""

from dataclasses import dataclass, field
import math

from strategy.base import StrategyDecision


@dataclass
class StrategyHistory:
    transitions: list[tuple[float, str]] = field(default_factory=list)
    targets: list[tuple[float, float]] = field(default_factory=list)
    last_decision: StrategyDecision | None = None

    def record(self, time_seconds: float, decision: StrategyDecision) -> None:
        if not self.transitions or decision.mode != self.transitions[-1][1]:
            self.transitions.append((time_seconds, decision.mode))
        if not self.targets or not math.isclose(decision.target_d, self.targets[-1][1]):
            self.targets.append((time_seconds, decision.target_d))
        self.last_decision = decision

    def print_summary(self, vehicle_id: str, strategy_name: str) -> None:
        print(f"{vehicle_id} strategy: {strategy_name}")
        print("Strategy transitions: " + " -> ".join(
            f"{time:.2f}s {mode.upper()}" for time, mode in self.transitions
        ))
        if strategy_name != "centerline":
            print("Lateral targets: " + " -> ".join(
                f"{time:.2f}s {target:+.2f}m" for time, target in self.targets
            ))
        if self.last_decision is not None:
            print(f"Final strategy reason: {self.last_decision.reason}")
        if strategy_name == "overtake":
            passed = any(mode == "return" for _, mode in self.transitions)
            returned = passed and self.transitions[-1][1] == "follow"
            print("Pass completed: " + ("yes" if passed else "no"))
            print("Return completed: " + ("yes" if returned else "no"))
