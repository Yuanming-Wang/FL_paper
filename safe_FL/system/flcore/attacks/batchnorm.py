"""BatchNorm policy shared by gradient inversion attacks."""

from torch import nn


def configure_attack_batchnorm(model, mode):
    """Use candidate batch statistics without importing private client buffers."""
    if mode not in ("running", "batch"):
        raise ValueError("batchnorm_mode must be 'running' or 'batch'")
    if mode == "batch":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
