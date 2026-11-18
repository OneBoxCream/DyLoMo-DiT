"""The weight-shared loop: which layers a forward pass visits, and how many passes a noise level gets.

The ``L`` physical layers form a front, a middle and a back segment. A forward pass runs the front segment, the middle
segment ``r`` times with the same weights, and the back segment; inside the middle segment every sub-layer writes to the
residual with a factor ``1 / r``. The number of passes ``r`` is a function of the noise level and is part of the
model: sampling must use the schedule the model was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


def layer_schedule(num_layers: int, loop_span: Optional[Sequence[int]], passes: int) -> List[Tuple[int, float]]:
    """``[(layer index, residual scale)]`` for one forward pass. ``loop_span`` is half-open, 0-based."""
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
        raise ValueError("the number of passes must be a positive integer")
    if loop_span is None:
        if passes != 1:
            raise ValueError("more than one pass needs a looped segment")
        return [(i, 1.0) for i in range(num_layers)]
    lo, hi = loop_span
    if not 0 <= lo < hi <= num_layers:
        raise ValueError("the looped segment must be a non-empty half-open range of layers")
    return ([(i, 1.0) for i in range(lo)]
            + [(i, 1.0 / passes) for _ in range(passes) for i in range(lo, hi)]
            + [(i, 1.0) for i in range(hi, num_layers)])


@dataclass(frozen=True)
class PassSchedule:
    """Number of passes through the looped segment as a function of the diffusion timestep ``t`` (0 .. 999).

    ``thresholds`` are ascending timesteps and ``passes`` has one more entry: ``t < thresholds[0]`` uses ``passes[0]``,
    ``thresholds[i-1] <= t < thresholds[i]`` uses ``passes[i]`` and ``t >= thresholds[-1]`` uses ``passes[-1]``.
    The released class-conditional models use ``thresholds=(500,)``, ``passes=(3, 1)``: three passes at low noise
    (``t < 500``) and one at high noise (``t >= 500``). An empty ``thresholds`` is a fixed number of passes.
    """

    thresholds: Tuple[int, ...] = ()
    passes: Tuple[int, ...] = (1,)

    def __post_init__(self):
        thresholds, passes = tuple(self.thresholds), tuple(self.passes)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "passes", passes)
        if len(passes) != len(thresholds) + 1:
            raise ValueError("need exactly one more pass count than thresholds")
        if any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in passes):
            raise ValueError("pass counts must be positive integers")
        if any(isinstance(t, bool) or not isinstance(t, int) for t in thresholds):
            raise ValueError("thresholds must be integers")
        if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError("thresholds must be strictly ascending")

    def passes_at(self, t: int) -> int:
        return self.passes[sum(1 for threshold in self.thresholds if t >= threshold)]

    @property
    def max_passes(self) -> int:
        return max(self.passes)

    def mean_passes(self, timesteps: Sequence[int]) -> float:
        """Average number of passes over a set of sampling timesteps."""
        return sum(self.passes_at(int(t)) for t in timesteps) / len(timesteps)

    @staticmethod
    def from_config(value) -> "PassSchedule":
        if isinstance(value, PassSchedule):
            return value
        if isinstance(value, int):
            return PassSchedule((), (value,))
        return PassSchedule(tuple(value.get("thresholds", ())), tuple(value["passes"]))

    def to_config(self) -> dict:
        return {"thresholds": list(self.thresholds), "passes": list(self.passes)}
