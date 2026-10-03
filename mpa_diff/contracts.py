from dataclasses import dataclass
from typing import Optional, List, Dict
import torch

@dataclass
class UIEBatch:
    image: torch.Tensor
    sample_ids: List[str]
    reference: Optional[torch.Tensor] = None
    valid_mask: Optional[torch.Tensor] = None
    metadata: Optional[List[dict]] = None

@dataclass
class DepthPrior:
    raw: torch.Tensor
    distance_proxy: torch.Tensor
    physical_coordinate: torch.Tensor
    confidence: torch.Tensor
    valid_mask: torch.Tensor
    metadata: dict
    log_variance: Optional[torch.Tensor] = None

@dataclass
class PriorBundle:
    image: torch.Tensor
    depth: DepthPrior
    physical_image: torch.Tensor
    histogram: torch.Tensor
    edges: torch.Tensor
    wavelet_high: torch.Tensor
    fields: Dict[str, torch.Tensor]
    metadata: dict
