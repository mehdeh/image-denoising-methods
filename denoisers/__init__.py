"""
Denoiser modules for various denoising methods.

This package contains different denoising approaches:
- ideal_denoiser: Theoretical ideal denoiser (Equation 57 from EDM paper)
- edm_denoiser: EDM neural network-based denoiser
"""

from .ideal_denoiser import ideal_denoiser
from .edm_denoiser import (
    load_edm_model,
    edm_denoise,
    compute_score_gradient,
    gradient_ascent_denoise,
    load_pretrained_edm
)

__all__ = [
    'ideal_denoiser',
    'load_edm_model',
    'edm_denoise',
    'compute_score_gradient',
    'gradient_ascent_denoise',
    'load_pretrained_edm'
]

