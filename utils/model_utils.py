"""
Model utilities for downloading and managing pretrained models.

This module provides utilities for downloading pretrained models from URLs
and managing model checkpoints.
"""

import os
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    
    def update_to(self, blocks=1, block_size=1, total_size=None):
        """Update progress bar for urllib.request.urlretrieve."""
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, destination: str) -> None:
    """
    Download a file from URL with progress bar.
    
    Parameters:
    -----------
    url : str
        URL to download from
    destination : str
        Local path to save the downloaded file
        
    Examples:
    ---------
    >>> from utils.model_utils import download_file
    >>> 
    >>> url = "https://example.com/model.pkl"
    >>> dest = "./models/model.pkl"
    >>> download_file(url, dest)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading from {url}")
    print(f"Saving to {destination}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Download") as progress_bar:
        urllib.request.urlretrieve(
            url,
            destination,
            reporthook=progress_bar.update_to
        )
    
    print(f"✓ Download completed: {destination}")


def ensure_model_downloaded(model_path: str, url: str) -> bool:
    """
    Ensure a model file exists, downloading if necessary.
    
    Parameters:
    -----------
    model_path : str
        Local path where model should exist
    url : str
        URL to download from if model doesn't exist
        
    Returns:
    --------
    success : bool
        True if model exists or was successfully downloaded
        
    Examples:
    ---------
    >>> from utils.model_utils import ensure_model_downloaded
    >>> 
    >>> model_path = "./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl"
    >>> url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl"
    >>> 
    >>> if ensure_model_downloaded(model_path, url):
    >>>     print("Model ready to use")
    """
    if os.path.exists(model_path):
        print(f"✓ Model found at: {model_path}")
        return True
    
    print(f"Model not found at: {model_path}")
    
    try:
        download_file(url, model_path)
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False

