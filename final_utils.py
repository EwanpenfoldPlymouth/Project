import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images, nrow=8, figsize=(8, 8)):
    """
    Plots a grid of images using matplotlib.
    
    Args:
        images (Tensor): Batch of images with shape (B, C, H, W).
        nrow (int): Number of images per row.
        figsize (tuple): Figure size.
    """
    grid_img = torchvision.utils.make_grid(images.cpu(), nrow=nrow, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=figsize)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


def save_images(images, path, nrow=8, padding=2, normalize=False, value_range=(0,255)):
    """
    Saves a batch of images in a grid to the specified file path.
    
    Since the images are already in uint8 [0,255], we set normalize=False
    by default and use a value_range of (0,255).
    """
    grid = torchvision.utils.make_grid(
        images, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range
    )
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)



class CustomDataset(torchvision.datasets.ImageFolder):
    """
    A custom ImageFolder dataset that limits the number of images to subset_size.
    """
    def __init__(self, root, transform=None, subset_size=100):
        super(CustomDataset, self).__init__(root, transform)
        self.subset_size = min(subset_size, len(self.samples))
        self.samples = self.samples[:self.subset_size]  # Limit to first subset_size images

    def __len__(self):
        return self.subset_size


def get_data(args):
    """
    Creates and returns a DataLoader for the dataset.
    
    The transformations include a dynamic resize, random crop, normalization, 
    and conversion to a tensor.
    
    Args:
        args: A namespace or dict with keys: dataset_path, image_size, batch_size, subset_size.
    
    Returns:
        DataLoader for the dataset.
    """
    # Compute resize size as a function of args.image_size (e.g., 1.25 times larger)
    resize_size = int(args.image_size * 1.25)
    transform_pipeline = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize_size),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset(root=args.dataset_path, transform=transform_pipeline, subset_size=args.subset_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloader


def setup_logging(run_name):
    """
    Creates required directories for logging models and results.
    
    Args:
        run_name (str): Identifier for the current run/experiment.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

