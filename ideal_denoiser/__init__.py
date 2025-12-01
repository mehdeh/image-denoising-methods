"""
Ideal Denoiser Package (Equation 57 from EDM Paper).

This package implements the theoretical ideal denoiser from the paper:
"Elucidating the Design Space of Diffusion-Based Generative Models"
by Karras et al., NeurIPS 2022.

Usage:
    from ideal_denoiser import ideal_denoiser
"""

from .core import ideal_denoiser

__all__ = ["ideal_denoiser"]
