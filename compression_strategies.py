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
    """True 1-bit sign compression with magnitude scaling."""
    def __init__(self, scale: str = "median"):
        # scale ∈ {"mean","median"}; median is more robust
        self.scale = scale

    def compress(self, tensor: torch.Tensor):
        # Flatten and compute scale
        flat = tensor.view(-1).detach()
        if flat.numel() == 0:
            packed = np.frombuffer(b"", dtype=np.uint8)
            meta = (flat.numel(), 0.0)
            return (packed, meta), tensor.shape

        if self.scale == "median":
            mag = flat.abs().median().item()
        else:
            mag = flat.abs().mean().item()
        # Avoid zero scaling
        if mag == 0.0:
            mag = 1e-8

        # Pack signs into bits: 1 -> 1, -1/0 -> 0
        # (You can choose >=0 as 1 to match your direction convention)
        bits = (flat >= 0).to(torch.uint8).cpu().numpy()  # 0/1 per element
        packed = np.packbits(bits)  # uint8 array, length = ceil(N/8)

        # meta carries (num_elements, magnitude)
        meta = (int(flat.numel()), float(mag))
        return (packed, meta), tensor.shape

    def decompress(self, compressed, shape):
        packed, meta = compressed
        n, mag = meta
        if n == 0:
            return torch.zeros(shape, dtype=torch.float32)

        # Unpack bits back to 0/1 then map to {-1,+1}
        bits = np.unpackbits(packed)[:n]
        signs = (bits * 2 - 1).astype(np.int8)  # 0->-1, 1->+1
        out = torch.from_numpy(signs).to(torch.float32) * float(mag)
        return out.view(shape)


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
