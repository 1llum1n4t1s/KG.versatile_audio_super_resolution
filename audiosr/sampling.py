"""Sampler names shared by the model, the pipeline, and the CLI.

This module stays free of PyTorch so ``audiosr --help`` can list the available
samplers without importing the model stack.
"""

# ``ddpm`` ignores the requested step count and runs the full training schedule;
# every other sampler honours ``ddim_steps``.
#
# ``ddim`` reproduces the established schedule, where ``ddim_eta=1.0`` makes each
# update fully ancestral. ``dpmpp2m`` integrates the deterministic
# probability-flow ODE instead, which reaches a comparable result in far fewer
# steps and ignores ``ddim_eta``.
SUPPORTED_SAMPLERS = ("ddim", "dpmpp2m", "ddpm")
DEFAULT_SAMPLER = "ddim"

# ``uniform`` reproduces the established schedule. It reaches the noisiest
# timestep only when the step count does not divide the training schedule; for
# exact divisors it stops short, which leaves a signal component that sampling
# from pure noise never accounts for. ``trailing`` always ends at the noisiest
# timestep. ``quad`` is the upstream quadratic spacing.
SUPPORTED_DISCRETIZATIONS = ("uniform", "trailing", "quad")
DEFAULT_DISCRETIZATION = "uniform"


def normalize_sampler(sampler):
    """Return the canonical name for ``sampler`` or raise ``ValueError``."""
    normalized = str(sampler).lower()
    if normalized not in SUPPORTED_SAMPLERS:
        raise ValueError(
            f"sampler must be one of {list(SUPPORTED_SAMPLERS)}, got {sampler!r}"
        )
    return normalized


def normalize_discretize(discretize):
    """Return the canonical timestep spacing name or raise ``ValueError``."""
    normalized = str(discretize).lower()
    if normalized not in SUPPORTED_DISCRETIZATIONS:
        raise ValueError(
            f"discretize must be one of {list(SUPPORTED_DISCRETIZATIONS)}, got "
            f"{discretize!r}"
        )
    return normalized


def normalize_ddim_eta(ddim_eta):
    """Return ``ddim_eta`` as a validated float.

    Only the ``ddim`` sampler consumes it; ``dpmpp2m`` is deterministic and
    ``ddpm`` follows the training schedule.
    """
    value = float(ddim_eta)
    if not value == value or value in (float("inf"), float("-inf")):
        raise ValueError("ddim_eta must be finite")
    if value < 0.0:
        raise ValueError("ddim_eta must not be negative")
    return value
