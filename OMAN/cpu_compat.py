import torch


def enable_cpu_compat_if_needed() -> torch.device:
    """
    Enable CPU fallback for code paths that hardcode `.cuda()` calls.
    Returns the effective torch.device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Monkey-patch Tensor/Module cuda() to no-op on CPU-only machines.
    # This keeps upstream repo code runnable without invasive edits.
    def _tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        return self.to("cpu")

    def _module_cuda(self, device=None):
        return self.to("cpu")

    torch.Tensor.cuda = _tensor_cuda
    torch.nn.Module.cuda = _module_cuda
    return torch.device("cpu")
