"""SIMD-aware retained-channel calculations."""


SUPPORTED_ALIGNMENTS = (1, 2, 4, 8, 16, 32)


def resolve_alignment(width: int, requested_alignment: int) -> int:
    """Select the largest usable alignment with at least two channel groups."""

    if width <= 0:
        raise ValueError("width must be positive")
    if requested_alignment not in SUPPORTED_ALIGNMENTS:
        raise ValueError(f"requested_alignment must be one of {SUPPORTED_ALIGNMENTS}")
    candidates = [requested_alignment // (2**power) for power in range(6)]
    candidates = [candidate for candidate in candidates if candidate >= 1]
    return next((candidate for candidate in candidates if width // candidate >= 2), 1)


def aligned_retained_width(
    width: int,
    pruning_ratio: float,
    requested_alignment: int,
    maximum_ratio: float = 0.9,
) -> tuple[int, int, float]:
    """Return retained width, effective alignment, and effective pruning ratio."""

    if width <= 0:
        raise ValueError("width must be positive")
    if not 0 <= pruning_ratio <= 1:
        raise ValueError("pruning_ratio must be between 0 and 1")
    if not 0 < maximum_ratio <= 1:
        raise ValueError("maximum_ratio must be between 0 and 1")
    ratio = min(pruning_ratio, maximum_ratio)
    alignment = resolve_alignment(width, requested_alignment)
    if alignment == 1 and width < 2:
        retained = 1
    else:
        groups = max(1, round(width * (1 - ratio) / alignment))
        retained = min(width, max(1, groups * alignment))
        if retained == 0:
            retained = 1
    effective_ratio = 1.0 - retained / width
    return retained, alignment, effective_ratio