"""Type stubs for the HOLA optimization engine."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__: list[str]

class Real:
    """Real-valued parameter with configurable scale.

    Args:
        min: Lower bound (in actual values, not exponents).
        max: Upper bound (in actual values, not exponents).
        scale: Sampling scale -- ``"linear"`` (default), ``"log"`` (natural log),
            or ``"log10"``.
    """

    min: float
    max: float
    scale: str
    def __init__(self, min: float, max: float, scale: str = "linear") -> None: ...

class Integer:
    """Integer parameter within an inclusive range.

    Args:
        min: Lower bound (inclusive).
        max: Upper bound (inclusive).
    """

    min: int
    max: int
    def __init__(self, min: int, max: int) -> None: ...

class Categorical:
    """Categorical parameter (choose from a list of string labels)."""

    choices: list[str]
    def __init__(self, choices: list[str]) -> None: ...

class Minimize:
    """Minimize an objective field using TLP (Target-Limit-Priority) scoring.

    Args:
        field: Name of the metric to minimize (must appear in the dict returned
            by the objective function).
        target: "Good enough" value -- at or below this, the TLP score is 0.
        limit: "Unacceptable" value -- beyond this, the trial is infeasible
            (score = inf).
        priority: Per-objective weight/slope in the TLP formula. Default 1.0.
        group: Priority-group label. Objectives sharing the same group are summed
            into one component of the group-cost vector for Pareto ranking. When
            omitted, defaults to the field name (one group per objective).
    """

    field: str
    target: float | None
    limit: float | None
    priority: float
    group: str | None
    def __init__(
        self,
        field: str,
        target: float | None = None,
        limit: float | None = None,
        priority: float = 1.0,
        group: str | None = None,
    ) -> None: ...

class Maximize:
    """Maximize an objective field using TLP (Target-Limit-Priority) scoring.

    Args:
        field: Name of the metric to maximize (must appear in the dict returned
            by the objective function).
        target: "Good enough" value -- at or above this, the TLP score is 0.
        limit: "Unacceptable" value -- below this, the trial is infeasible
            (score = inf).
        priority: Per-objective weight/slope in the TLP formula. Default 1.0.
        group: Priority-group label. See ``Minimize`` for details.
    """

    field: str
    target: float | None
    limit: float | None
    priority: float
    group: str | None
    def __init__(
        self,
        field: str,
        target: float | None = None,
        limit: float | None = None,
        priority: float = 1.0,
        group: str | None = None,
    ) -> None: ...

class Gmm:
    """GMM strategy configuration.

    Configures the Gaussian Mixture Model strategy. Use this class instead of
    the string ``"gmm"`` when you need to adjust refit behavior.

    Args:
        refit_interval: How often the GMM is refit, in completed trials
            (default: 20).
        elite_fraction: Fraction of top trials used for GMM refitting
            (default: 0.25). Must be between 0.0 and 1.0.
        exploration_budget: Number of Sobol exploration trials before GMM
            exploitation begins. When omitted, computed automatically from
            the number of dimensions.
    """

    refit_interval: int | None
    elite_fraction: float | None
    exploration_budget: int | None
    def __init__(
        self,
        refit_interval: int | None = None,
        elite_fraction: float | None = None,
        exploration_budget: int | None = None,
    ) -> None: ...

class Sobol:
    """Sobol strategy configuration.

    Owen-scrambled Sobol quasi-random sequences provide better space coverage
    than pure random sampling.
    """

    def __init__(self) -> None: ...

class Random:
    """Random strategy configuration.

    Uniform pseudo-random sampling.
    """

    def __init__(self) -> None: ...

class Space:
    """Named parameter space builder.

    Pass parameter definitions as keyword arguments::

        Space(lr=Real(1e-4, 0.1, scale="log10"), layers=Integer(1, 10))
    """

    def __init__(self, **kwargs: Real | Integer | Categorical) -> None: ...

class Trial:
    """A trial returned by ``Study.ask()``."""

    trial_id: int
    params: dict[str, Any]
    def __repr__(self) -> str: ...

class CompletedTrial:
    """A completed trial with scoring, ranking, and Pareto front information."""

    trial_id: int
    params: dict[str, Any]
    metrics: dict[str, Any]
    scores: dict[str, Any]
    score_vector: dict[str, float]
    rank: int
    pareto_front: int
    completed_at: int
    def __repr__(self) -> str: ...

class Study:
    """The main optimization study.

    Example::

        study = Study(
            space=Space(lr=Real(1e-4, 0.1, scale="log10")),
            objectives=[Minimize("loss")],
        )
        trial = study.ask()
        ct = study.tell(trial.trial_id, {"loss": 0.42})
        top = study.top_k(3)
    """

    def __init__(
        self,
        space: Space,
        objectives: list[Minimize | Maximize],
        strategy: str | Gmm | Sobol | Random | None = None,
        seed: int | None = None,
        max_trials: int | None = None,
    ) -> None: ...
    @staticmethod
    def connect(url: str) -> Study:
        """Connect to an existing HOLA server."""
        ...
    @staticmethod
    def load(path: str) -> Study:
        """Load a study from a saved checkpoint."""
        ...
    def ask(self) -> Trial:
        """Request the next trial to evaluate."""
        ...
    def tell(self, trial_id: int, metrics: dict[str, Any]) -> CompletedTrial:
        """Report the result of a trial."""
        ...
    def cancel(self, trial_id: int) -> None:
        """Cancel a pending trial."""
        ...
    def top_k(self, k: int, include_infeasible: bool = False) -> list[CompletedTrial]:
        """Get the top-k trials by rank."""
        ...
    def pareto_front(
        self, front: int = 0, include_infeasible: bool = False
    ) -> list[CompletedTrial]:
        """Get trials on a specific Pareto front."""
        ...
    def trials(
        self, sorted_by: str = "index", include_infeasible: bool = True
    ) -> list[CompletedTrial]:
        """Get all trials with scoring and ranking."""
        ...
    def trial_count(self) -> int:
        """Number of completed trials."""
        ...
    def update_objectives(self, objectives: list[Minimize | Maximize]) -> None:
        """Update objectives mid-run, re-scalarizing all trials."""
        ...
    def save(self, path: str) -> None:
        """Save a checkpoint to disk."""
        ...
    def run(
        self,
        func: Callable[[dict[str, Any]], dict[str, Any]],
        n_trials: int,
        n_workers: int = 1,
    ) -> Study:
        """Run an objective function for n_trials, automating the ask/tell loop.

        Returns self so you can chain: ``study.run(func, 100).top_k(3)``
        """
        ...
    def serve(
        self,
        port: int = 8000,
        background: bool = False,
        dashboard_path: str | None = None,
    ) -> None:
        """Start a REST server for this study."""
        ...

def dashboard_dir() -> Path:
    """Return the path to the bundled dashboard static files.

    Useful for passing to ``Study.serve(dashboard_path=str(dashboard_dir()))``.
    Returns a ``pathlib.Path`` even if the directory does not exist (the dashboard
    is only bundled in release wheels, not editable installs from source).
    """
    ...
