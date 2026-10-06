"""Inference-only memory layout; training streams and parameter values are unchanged."""
import torch


INFERENCE_RUNTIME = {
    'version': 'channels_last_deterministic_v1',
    'memory_format': 'channels_last',
    'cudnn_benchmark': False,
    'cudnn_deterministic': True,
    'precision': 'float32_no_AMP',
    'batch': 1,
}


def inference_model(model):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return model.to(memory_format=torch.channels_last).eval()
