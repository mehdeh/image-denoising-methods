import sys
import os

# Get the project's root directory 
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("project_dir:", project_dir)

# Add EDM dependencies to path
edm_path = os.path.join(project_dir, "denoisers", "edm")
if edm_path not in sys.path:
    sys.path.insert(0, edm_path)

os.chdir(project_dir)

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import pickle
import numpy as np
import PIL.Image
import dnnlib
import matplotlib.pyplot as plt


##################################################
# Helper functions
##################################################

def normalize_batch_01(x: torch.Tensor):
    """
    Min-max normalization for a batch of images so that each image
    is scaled between 0 and 1 independently.

    Parameters:
    -----------
    x : torch.Tensor
        Input images of shape (batch_size, channels, height, width)

    Returns:
    --------
    x_normed : torch.Tensor
        Min-max normalized images with the same shape as x.
    """
    batch_size = x.shape[0]
    x_reshaped = x.view(batch_size, -1)  # Flatten each image
    min_vals = x_reshaped.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    max_vals = x_reshaped.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    return (x - min_vals) / (max_vals - min_vals + 1e-8)


def display_or_save_images(imgs: torch.Tensor, nrow=4, save_path=None):
    """
    Displays a batch of images in a grid. If 'save_path' is provided,
    saves the generated image instead of displaying.

    Parameters:
    -----------
    imgs : torch.Tensor
        A batch of images of shape (batch_size, channels, height, width).
    nrow : int, optional
        Number of images per row in the grid.
    save_path : str or None
        If provided, the image is saved to this path instead of showing.
    """
    img_grid = make_grid(imgs, nrow=nrow, padding=1)
    img_np = img_grid.permute(1, 2, 0).cpu().numpy()  # Convert to NumPy

    plt.figure(figsize=(2, 2))
    # Normalize for visualization (0..1) just in display
    plt.imshow((img_np - img_np.min()) / (img_np.max() - img_np.min()))
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
        print(f"Saved image to {save_path}")
    else:
        plt.show()
    plt.close()

def load_denoise_model(model_path: str, url: str, device: torch.device = torch.device('cuda')):
    """
    Loads a pretrained denoising model from a given path. If the model file does not exist,
    it downloads the file from the given URL and saves it to the specified path.

    Parameters:
    -----------
    model_path : str
        Local path where the model file should be stored and loaded from.
    url : str
        URL to download the model file if it does not already exist.
    device : torch.device
        Device to which the model will be moved.

    Returns:
    --------
    net : torch.nn.Module
        Loaded denoising model.
    """
    
    # Check if the model file already exists
    if os.path.exists(model_path):
        print(f"Loading model from local path: {model_path}")
        with open(model_path, "rb") as f:
            net = pickle.load(f)['ema'].to(device)
    else:
        print(f"Model not found at {model_path}. Downloading from {url}...")
        with dnnlib.util.open_url(url) as f:
            net = pickle.load(f)['ema'].to(device)
        
        # Save the downloaded model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
        with open(model_path, "wb") as f:
            pickle.dump({'ema': net}, f)  # Save model to local path
        print(f"Model downloaded and saved at {model_path}")

    return net

def load_cifar10_subset(root="./sigma_model/data", train=False, selected_indices=[20, 21, 22, 23]):
    """
    Loads a subset of CIFAR-10 dataset with given indices.

    Parameters:
    -----------
    root : str
        Directory where CIFAR-10 dataset is stored.
    train : bool
        Whether to load training data (True) or test data (False).
    selected_indices : list of int
        List of indices to select specific images from the dataset.

    Returns:
    --------
    inputs : torch.Tensor
        A batch of selected CIFAR-10 images.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform
    )

    inputs = torch.stack([dataset[i][0] for i in selected_indices])

    print(f"Loaded test images shape: {inputs.shape}")  # Expect (4, 3, 32, 32)
    return inputs


def add_noise_to_inputs(inputs, sigma_value=3.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Adds Gaussian noise to input images.

    Parameters:
    -----------
    inputs : torch.Tensor
        Clean images (batch of images).
    sigma_value : float
        The standard deviation of the Gaussian noise.
    device : torch.device
        The device (CPU/GPU) on which computations should run.

    Returns:
    --------
    noisy_inputs : torch.Tensor
        Noisy version of input images.
    sigma_ : torch.Tensor
        Standard deviation tensor for noise.
    """
    sigma_ = torch.tensor([sigma_value], dtype=torch.float64, device=device)
    eps = torch.randn_like(inputs).to(device)
    sigma_broadcast = sigma_[:, None, None, None]
    noisy_inputs = inputs + sigma_broadcast * eps

    # Convert to double precision
    noisy_inputs = noisy_inputs.to(torch.float64).to(device)
    
    return noisy_inputs, sigma_


##################################################
# Main gradient descent function
##################################################

def run_gradient_ascent(x_cur: torch.Tensor,
                        denoise_model: torch.nn.Module,
                        sigma_: torch.Tensor,
                        lr: float = 1.0,
                        num_iterations: int = 5):
    """
    Performs gradient ascent to move x_cur towards the maximum
    of the denoising model's PDF.

    Parameters:
    -----------
    x_cur : torch.Tensor
        Initial noisy images (batch of images).
    denoise_model : torch.nn.Module
        A pretrained denoising model.
    sigma_ : torch.Tensor
        Standard deviation parameter as a tensor (shape = [1]) or broadcastable shape.
    lr : float
        Learning rate for gradient updates.
    num_iterations : int
        Number of gradient ascent steps to perform.

    Returns:
    --------
    x_cur : torch.Tensor
        Updated images after all iterations.
    """
    for i in range(num_iterations):
        # 1) Pass current x through the denoising model
        net_denoised_img = denoise_model(x_cur, sigma_, class_labels=None).to(torch.float64)

        # 2) Compute gradient of log pdf
        # grad_x_log_pdf_value = -(x_cur - net_denoised_img) / sigma_^2
        grad_x_log_pdf_value = - (x_cur - net_denoised_img) / (sigma_ ** 2)

        # 3) Update x in the direction of the gradient (ascent)
        x_cur = x_cur + lr * grad_x_log_pdf_value

        # 4) Print results
        norm_values = torch.norm(grad_x_log_pdf_value.view(grad_x_log_pdf_value.shape[0], -1), dim=1)
        print(f"Iteration {i+1}:")
        print(f"  Norm2 of grad_x_log_pdf_value = {norm_values}")

        # Show updated images
        print("  Showing updated x_cur images:")
        display_or_save_images(normalize_batch_01(x_cur.detach()))

        # Show denoised images
        print("  Showing denoised x_cur images:")
        display_or_save_images(normalize_batch_01(net_denoised_img.detach()))

    return x_cur

##################################################
# Demo function (instead of __main__)
##################################################

def run_denoiser_demo():
    """
    This function demonstrates loading a small batch of CIFAR-10 images,
    adding noise, and performing gradient ascent to move the noisy images
    towards denoised images using a pretrained EDM model.

    You can import this function in a notebook or another script and call it:
    >>> from edm_denoiser_gradient import run_denoiser_demo
    >>> run_denoiser_demo()
    """

    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    # Define model storage path (local path)
    local_model_path = "./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl"  # مسیر دلخواه شما

    # Define the model's download URL
    denoise_model_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl"

    # Load or download the model
    denoise_model = load_denoise_model(local_model_path, denoise_model_url, device)

    # Load CIFAR-10 subset
    inputs = load_cifar10_subset()

    # Add noise to inputs
    x_cur, sigma_ = add_noise_to_inputs(inputs)
    
    # Show original images
    print("Original input (clean) images:")
    display_or_save_images(normalize_batch_01(inputs))

    # Show noisy inputs
    print("Noisy inputs:")
    display_or_save_images(normalize_batch_01(x_cur))

    # Run gradient ascent
    final_x = run_gradient_ascent(
        x_cur=x_cur,
        denoise_model=denoise_model,
        sigma_=sigma_,
        lr=1.0,
        num_iterations=10  # Number of steps
    )

    print("\nGradient ascent finished.")
    print("Showing final x_cur images:")
    display_or_save_images(normalize_batch_01(final_x.detach()))


# If you wanted to keep an __main__ block for direct CLI usage, you could do:
if __name__ == "__main__":
    run_denoiser_demo()


