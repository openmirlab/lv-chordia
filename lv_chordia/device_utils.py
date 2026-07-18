"""
Device-selection resolution: turns a caller-facing device string into the
use_gpu bool NetworkBehavior/ChordNet actually consume.

chord_recognition()'s optional device parameter accepts 'cpu', 'cuda',
'cuda:N', 'auto', or None. This is the one place that knows how to turn that
string into a bool (or None, to defer to NetworkBehavior's own
torch.cuda.device_count() > 0 auto-detect) -- NetworkBehavior/ChordNet stay
unaware of the string format, and chord_recognition.py/cli.py stay unaware of
how auto-detection works.

Reads: nothing (torch only).
"""

from typing import Optional

import torch


def resolve_use_gpu(device: Optional[str] = None) -> Optional[bool]:
    """
    Resolve a caller-facing device string into the use_gpu bool NetworkBehavior
    consumes.

    Args:
        device: One of 'cpu', 'cuda', 'cuda:N' (N a GPU index), 'auto', or
            None. None (the default) returns None, telling NetworkBehavior to
            keep auto-detecting via torch.cuda.device_count() > 0 exactly as
            it does today -- passing nothing preserves today's exact
            behavior. 'auto' requests that same auto-detect explicitly and
            also returns None. 'cpu' forces use_gpu=False even when CUDA is
            available. 'cuda' and 'cuda:N' force use_gpu=True, raising if no
            CUDA device is visible.

    Returns:
        True to force GPU, False to force CPU, or None to defer to
        NetworkBehavior's own auto-detection.

    Raises:
        RuntimeError: device requests CUDA ('cuda' or 'cuda:N') but no CUDA
            device is visible to torch, or requests an index N that is out
            of range for the visible devices.
        ValueError: device is not one of the recognized strings.

    Only GPU-yes/no is wired through the inference pipeline below this
    function -- every .cuda() call in NetworkBehavior/ChordNet targets
    torch's "current" CUDA device, not a caller-chosen index. 'cuda:N' is
    parsed and bounds-checked against torch.cuda.device_count() (so a typo
    like 'cuda:9' on a 1-GPU machine still fails loudly), but N does not
    select which physical GPU actually runs the model -- that still requires
    CUDA_VISIBLE_DEVICES set before the process starts, the same as today.
    """
    if device is None or device == 'auto':
        return None
    if device == 'cpu':
        return False
    if device == 'cuda':
        if torch.cuda.device_count() == 0:
            raise RuntimeError(
                "device='cuda' was requested but no CUDA device is visible "
                "(torch.cuda.device_count() == 0)."
            )
        return True
    if device.startswith('cuda:'):
        index_str = device[len('cuda:'):]
        if not index_str.isdigit():
            raise ValueError(
                f"Invalid device {device!r}: expected 'cuda:N' with N a "
                "non-negative integer GPU index."
            )
        index = int(index_str)
        available = torch.cuda.device_count()
        if available == 0 or index >= available:
            raise RuntimeError(
                f"device={device!r} was requested but only {available} CUDA "
                f"device(s) are visible (torch.cuda.device_count() == {available})."
            )
        return True
    raise ValueError(
        f"Invalid device {device!r}: expected one of 'cpu', 'cuda', 'cuda:N', "
        "'auto', or None."
    )
