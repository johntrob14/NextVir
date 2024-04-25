import torch
from torch import einsum
from typing import Union
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from einops import rearrange


# =================================================================== #

class LoRaModule(nn.Module):
    def __init__(self,
                 input_dimension: int, 
                 output_dimension: int,
                 split_dimension: int = 1,
                 device: torch.device = None,
                 rank: int = 1,
                 alpha: float = 1.
    ):
        super(LoRaModule, self).__init__()
        if split_dimension != 1: # Thank you pytorch for combining kqv into one param :(
            head_dimension = output_dimension // split_dimension
            assert head_dimension * split_dimension == output_dimension, \
                "Split dimension must be a multiple of input dimension"

            # Create's a lora for each projection matrix
            self.lora_A = nn.Parameter(torch.zeros(split_dimension, rank, head_dimension))
            self.lora_B = nn.Parameter(torch.zeros(split_dimension, input_dimension, rank))
            nn.init.normal_(self.lora_A, mean=0, std=1)
            self.split_dimension = split_dimension

        else:
            # Described in Section 4.1 of the LoRA paper
            self.lora_A = nn.Parameter(torch.zeros(rank, output_dimension))
            self.lora_B = nn.Parameter(torch.zeros(input_dimension, rank))
            nn.init.normal_(self.lora_A, mean=0, std=1)
            self.split_dimension = None

        # Described in Section 4.1 of the paper
        self.scale = alpha / rank

        self.enabled = True

        if device is not None:
            self.to(device)

    def forward(self, original_weights):
        if self.enabled:
            if self.split_dimension is None:
                return original_weights + (self.lora_B @ self.lora_A).view(original_weights.shape) * self.scale
            else:
                return original_weights + self.scale * \
                        rearrange(einsum('hir,hro->hio', self.lora_B, self.lora_A), 'h i o-> i (h o)').view(original_weights.shape)
        return original_weights


# =================================================================== #
