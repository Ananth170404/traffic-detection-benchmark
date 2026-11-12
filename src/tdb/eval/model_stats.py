"""Model size statistics: parameter count, FLOPs, weight file size."""

from pathlib import Path


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_flops(module, example_input) -> int | None:
    """Forward-pass FLOPs via torch's FlopCounterMode (counts a multiply-add as 2 FLOPs).

    Returns None if the model cannot be traced this way (e.g. non-torch models).
    """
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    module.eval()
    counter = FlopCounterMode(display=False)
    try:
        with torch.no_grad(), counter:
            module(example_input)
    except Exception:
        return None
    return int(counter.get_total_flops())


def file_size_mb(path: str | Path | None) -> float | None:
    if path is None or not Path(path).exists():
        return None
    return round(Path(path).stat().st_size / 2**20, 2)
