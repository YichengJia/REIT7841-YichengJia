"""
compression_strategies.py
Implements gradient compression strategies for federated learning.
"""

import torch
import numpy as np


class Compressor:
    """Base class for compressors"""
    def compress(self, tensor: torch.Tensor):
        raise NotImplementedError

    def decompress(self, compressed, shape):
        raise NotImplementedError


class TopKCompressor(Compressor):
    """Top-K sparsification"""
    def __init__(self, k: int):
        self.k = k

    def compress(self, tensor: torch.Tensor):
        tensor_flat = tensor.view(-1)
        k = min(self.k, tensor_flat.numel())

        # Get top-k indices
        _, indices = torch.topk(tensor_flat.abs(), k, sorted=False)
        values = tensor_flat[indices]

        return (indices.cpu().numpy(), values.cpu().numpy()), tensor.shape

    def decompress(self, compressed, shape):
        indices, values = compressed
        tensor_flat = torch.zeros(np.prod(shape), dtype=torch.float32)
        tensor_flat[indices] = torch.tensor(values, dtype=torch.float32)
        return tensor_flat.view(shape)


class SignSGDCompressor(Compressor):
    """SignSGD compression"""
    def compress(self, tensor: torch.Tensor):
        sign = torch.sign(tensor).cpu().numpy()
        return sign, tensor.shape

    def decompress(self, compressed, shape):
        sign = compressed
        return torch.tensor(sign, dtype=torch.float32).view(shape)


class QSGDCompressor(Compressor):
    """Quantized SGD"""
    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits

    def compress(self, tensor: torch.Tensor):
        tensor_flat = tensor.view(-1)
        norm = torch.norm(tensor_flat)

        if norm.item() == 0:
            return (np.zeros_like(tensor_flat.cpu().numpy()), 0.0, 1.0), tensor.shape

        q_levels = 2 ** self.num_bits
        scale = norm / q_levels
        scaled = torch.abs(tensor_flat) / scale
        probs = scaled - torch.floor(scaled)
        quantized = torch.floor(scaled) + torch.bernoulli(probs)
        signs = torch.sign(tensor_flat)

        values = (quantized * signs).cpu().numpy()
        return (values, norm.item(), scale.item()), tensor.shape

    def decompress(self, compressed, shape):
        values, norm, scale = compressed
        tensor_flat = torch.tensor(values, dtype=torch.float32) * scale
        return tensor_flat.view(shape)
