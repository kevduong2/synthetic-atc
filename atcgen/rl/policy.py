"""Optimizers over the unit cube, sized for a budget of tens of evaluations.

Every candidate costs a synthetic-batch generation plus a whisper-tiny
fine-tune — roughly ten minutes — so the realistic budget is 20-60 evaluations
for a ~19-dimensional space.  That rules out anything that needs a surrogate
model fit or a large population, and it makes the *baseline* matter: with so
few samples a search has to be shown to beat plain random sampling, which is
why :class:`RandomSearch` is a first-class member here rather than a strawman.

Three strategies, all numpy-only and all resumable:

*   :class:`RandomSearch` — the control.
*   :class:`ReinforceGaussian` — the AutoAugment-style controller, shrunk: a
    factored Gaussian trained by REINFORCE on standardized rewards.  Follows a
    gradient, so it keeps improving in one direction across iterations, but it
    is the noisier of the two learners at this budget.
*   :class:`CrossEntropyMethod` — fit to the elite fraction each round.  The
    robust default: it ignores reward *scale* entirely and only uses the
    ranking, which is the right assumption when each reward is a WER delta
    measured through a short, noisy fine-tune.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class Optimizer(Protocol):
    """Ask/tell interface over [0,1]^d.

    ``ask`` returns candidate vectors, ``tell`` reports their rewards (higher
    is better).  ``tell`` must tolerate batches of any size >= 1 and batches
    smaller than the last ``ask`` — the loop drops candidates whose evaluation
    raised.  State round-trips through plain JSON-serializable dicts.
    """

    def ask(self, n: int) -> list[np.ndarray]: ...

    def tell(self, vectors: list[np.ndarray], rewards: list[float]) -> None: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


def _standardize(rewards: Sequence[float]) -> np.ndarray:
    """Per-batch reward standardization, the advantage signal for REINFORCE.

    Centring makes the update indifferent to the reward's absolute level (WER
    deltas drift as the baseline model changes); dividing by the spread keeps
    the step size stable when one iteration happens to be flat.  A degenerate
    batch — one sample, or all rewards equal — yields all-zero advantages and
    therefore no update, which is the honest outcome.
    """
    values = np.asarray(rewards, dtype=float)
    spread = float(values.std())
    if spread < 1e-8:
        return np.zeros_like(values)
    return (values - values.mean()) / spread


def _rng_state(rng: np.random.Generator) -> dict:
    return {"bit_generator": rng.bit_generator.state}


def _rng_from_state(state: dict) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = state["bit_generator"]
    return rng


class RandomSearch:
    """Uniform sampling; ``tell`` is a no-op.

    The control arm.  At 20-60 evaluations a learner that cannot beat this is
    not paying for the extra machinery, and the loop's trial log makes the
    comparison directly.
    """

    def __init__(self, dim: int, seed: int = 0):
        self.dim = int(dim)
        self._rng = np.random.default_rng(seed)

    def ask(self, n: int) -> list[np.ndarray]:
        return [self._rng.random(self.dim) for _ in range(int(n))]

    def tell(self, vectors: list[np.ndarray], rewards: list[float]) -> None:
        return None

    def state_dict(self) -> dict:
        return {"kind": "random", "dim": self.dim, "rng": _rng_state(self._rng)}

    def load_state_dict(self, state: dict) -> None:
        self.dim = int(state["dim"])
        self._rng = _rng_from_state(state["rng"])


class ReinforceGaussian:
    """Factored Gaussian policy N(mean, diag(sigma^2)) trained by REINFORCE.

    Samples are drawn unclipped and then clipped into the cube for evaluation.
    The gradient uses the **pre-clip** sample: clipping is part of the
    environment, not the policy, and scoring the clipped value would tell the
    policy that a mean far outside the cube produced the boundary sample, which
    lets the mean drift arbitrarily far out.

    The analytic score functions for a diagonal Gaussian, with z = (x-mu)/sigma:

        d log pi / d mu        = z / sigma
        d log pi / d log sigma = z^2 - 1

    The mean update is preconditioned by ``sigma`` (i.e. it applies
    ``sigma * score``, the standard evolution-strategies form).  Without that
    factor the raw ``1/sigma`` makes the step size explode exactly when the
    policy has become confident and should be taking *small* steps.
    """

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        init_mean: np.ndarray | Sequence[float] | None = None,
        init_sigma: float = 0.15,
        lr_mean: float = 0.25,
        lr_sigma: float = 0.08,
        sigma_min: float = 0.03,
        sigma_max: float = 0.35,
    ):
        self.dim = int(dim)
        self._rng = np.random.default_rng(seed)
        self.lr_mean = float(lr_mean)
        self.lr_sigma = float(lr_sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        mean = np.full(self.dim, 0.5) if init_mean is None else np.asarray(init_mean, float)
        if mean.shape != (self.dim,):
            raise ValueError(f"init_mean must have {self.dim} entries")
        self.mean = np.clip(mean.astype(float), 0.0, 1.0)
        self.sigma = np.full(self.dim, float(init_sigma))
        np.clip(self.sigma, self.sigma_min, self.sigma_max, out=self.sigma)
        # Pre-clip samples of the outstanding batch, paired with what was
        # handed out, so ``tell`` can recover them by identity.
        self._pending: list[tuple[np.ndarray, np.ndarray]] = []

    def ask(self, n: int) -> list[np.ndarray]:
        """Draw ``n`` candidates.  Clears any outstanding batch first.

        The loop's contract is ask -> evaluate -> tell, so a fresh ``ask``
        means the previous batch will never be reported (a crash, or a
        resume); dropping it keeps stale pre-clip samples from being matched
        against a later batch.
        """
        self._pending = []
        out = []
        for _ in range(int(n)):
            raw = self.mean + self.sigma * self._rng.standard_normal(self.dim)
            clipped = np.clip(raw, 0.0, 1.0)
            self._pending.append((clipped, raw))
            out.append(clipped)
        return out

    def _raw_for(self, vector: np.ndarray) -> np.ndarray:
        """Recover the pre-clip sample matching ``vector``, consuming the pair."""
        for index, (clipped, raw) in enumerate(self._pending):
            if np.array_equal(clipped, vector):
                self._pending.pop(index)
                return raw
        return np.asarray(vector, dtype=float)

    def tell(self, vectors: list[np.ndarray], rewards: list[float]) -> None:
        if len(vectors) != len(rewards):
            raise ValueError("vectors and rewards must have the same length")
        if not vectors:
            return
        advantages = _standardize(rewards)
        if not np.any(advantages):
            self._pending = []
            return

        samples = np.stack([self._raw_for(np.asarray(v, dtype=float)) for v in vectors])
        z = (samples - self.mean) / self.sigma

        self.mean = np.clip(
            self.mean + self.lr_mean * (advantages[:, None] * z).mean(axis=0), 0.0, 1.0)
        log_sigma = np.log(self.sigma) + self.lr_sigma * (
            advantages[:, None] * (z ** 2 - 1.0)).mean(axis=0)
        self.sigma = np.clip(np.exp(log_sigma), self.sigma_min, self.sigma_max)
        self._pending = []

    def state_dict(self) -> dict:
        return {
            "kind": "reinforce",
            "dim": self.dim,
            "mean": self.mean.tolist(),
            "sigma": self.sigma.tolist(),
            "lr_mean": self.lr_mean,
            "lr_sigma": self.lr_sigma,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "pending": [[c.tolist(), r.tolist()] for c, r in self._pending],
            "rng": _rng_state(self._rng),
        }

    def load_state_dict(self, state: dict) -> None:
        self.dim = int(state["dim"])
        self.mean = np.asarray(state["mean"], dtype=float)
        self.sigma = np.asarray(state["sigma"], dtype=float)
        self.lr_mean = float(state["lr_mean"])
        self.lr_sigma = float(state["lr_sigma"])
        self.sigma_min = float(state["sigma_min"])
        self.sigma_max = float(state["sigma_max"])
        self._pending = [
            (np.asarray(clipped, dtype=float), np.asarray(raw, dtype=float))
            for clipped, raw in state.get("pending", [])
        ]
        self._rng = _rng_from_state(state["rng"])


class CrossEntropyMethod:
    """Refit a diagonal Gaussian to the top ``pop_frac_elite`` each round.

    Rank-based, so a single wild reward cannot dominate an update the way it
    can with REINFORCE — the reason this is the recommended default here.  Two
    guards make it survive tiny batches: elite parameters are blended with the
    previous ones (``smoothing`` weights the fresh fit), and sigma never falls
    below ``sigma_decay_floor``, without which a batch of near-identical elites
    collapses the search on iteration two and never recovers.

    Unlike :class:`ReinforceGaussian` this fits to the *clipped* vectors that
    were actually evaluated; a distribution fitted to elites is a description
    of good points, and points outside the cube are not points at all.
    """

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        pop_frac_elite: float = 0.4,
        init_mean: np.ndarray | Sequence[float] | None = None,
        init_sigma: float = 0.25,
        sigma_decay_floor: float = 0.05,
        smoothing: float = 0.7,
    ):
        self.dim = int(dim)
        self._rng = np.random.default_rng(seed)
        self.pop_frac_elite = float(pop_frac_elite)
        self.sigma_decay_floor = float(sigma_decay_floor)
        self.smoothing = float(smoothing)
        mean = np.full(self.dim, 0.5) if init_mean is None else np.asarray(init_mean, float)
        if mean.shape != (self.dim,):
            raise ValueError(f"init_mean must have {self.dim} entries")
        self.mean = np.clip(mean.astype(float), 0.0, 1.0)
        self.sigma = np.full(self.dim, float(init_sigma))

    def ask(self, n: int) -> list[np.ndarray]:
        return [
            np.clip(self.mean + self.sigma * self._rng.standard_normal(self.dim), 0.0, 1.0)
            for _ in range(int(n))
        ]

    def tell(self, vectors: list[np.ndarray], rewards: list[float]) -> None:
        if len(vectors) != len(rewards):
            raise ValueError("vectors and rewards must have the same length")
        if not vectors:
            return
        samples = np.stack([np.asarray(v, dtype=float) for v in vectors])
        values = np.asarray(rewards, dtype=float)
        count = max(1, min(len(values), math.ceil(self.pop_frac_elite * len(values))))
        elite = samples[np.argsort(-values)[:count]]

        fit_mean = elite.mean(axis=0)
        fit_sigma = elite.std(axis=0) if count > 1 else self.sigma
        blend = self.smoothing
        self.mean = np.clip(blend * fit_mean + (1.0 - blend) * self.mean, 0.0, 1.0)
        self.sigma = np.maximum(
            blend * fit_sigma + (1.0 - blend) * self.sigma, self.sigma_decay_floor)

    def state_dict(self) -> dict:
        return {
            "kind": "cem",
            "dim": self.dim,
            "mean": self.mean.tolist(),
            "sigma": self.sigma.tolist(),
            "pop_frac_elite": self.pop_frac_elite,
            "sigma_decay_floor": self.sigma_decay_floor,
            "smoothing": self.smoothing,
            "rng": _rng_state(self._rng),
        }

    def load_state_dict(self, state: dict) -> None:
        self.dim = int(state["dim"])
        self.mean = np.asarray(state["mean"], dtype=float)
        self.sigma = np.asarray(state["sigma"], dtype=float)
        self.pop_frac_elite = float(state["pop_frac_elite"])
        self.sigma_decay_floor = float(state["sigma_decay_floor"])
        self.smoothing = float(state["smoothing"])
        self._rng = _rng_from_state(state["rng"])


OPTIMIZERS: dict[str, Any] = {
    "random": RandomSearch,
    "reinforce": ReinforceGaussian,
    "cem": CrossEntropyMethod,
}
"""CLI name -> class, so scripts do not carry their own dispatch table."""
