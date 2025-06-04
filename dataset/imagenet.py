import os
import shutil
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from typing import Tuple
import requests
import zipfile


__all__ = ["get_imagenet_dataloaders"]


def _is_dataset_valid():
    """Check if the dataset exists and is in the correct format."""
    dataset_path = "tiny-imagenet/tiny-imagenet-200"
    val_annotations_file = os.path.join(dataset_path, "val/val_annotations.txt")
    images_folder = os.path.join(dataset_path, "val/images")

    # Check if the required files and folders exist
    if (
        os.path.exists(dataset_path)
        and os.path.exists(val_annotations_file)
        and os.path.exists(images_folder)
    ):
        return True
    return False


def _download_and_extract():
    """Download and unzip the Tiny ImageNet dataset if not already present."""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset_path = "tiny-imagenet"
    zip_path = "tiny-imagenet-200.zip"

    # If dataset directory doesn't exist, download and unzip the dataset
    if not os.path.exists(dataset_path):
        # Download the file
        print("Downloading Tiny ImageNet dataset...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download completed.")

        # Unzip the downloaded file
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)
        print("Extraction completed.")

        # Remove the zip file after extraction
        os.remove(zip_path)
    else:
        print(f"{dataset_path} already exists. Skipping download and extraction.")


def _adjust_validation_format():
    """Adjust the format of the validation set to match ImageFolder format."""
    val_annotations_file = "tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt"

    if not os.path.exists(val_annotations_file):
        raise FileNotFoundError(f"{val_annotations_file} not found.")

    # Check if the images are already in the correct format
    images_folder = "tiny-imagenet/tiny-imagenet-200/val/images"
    if not os.listdir(images_folder):
        print(f"Images folder {images_folder} is empty. Skipping reformatting.")
        return

    with open(val_annotations_file) as f:
        for line in f:
            fn, cls, *_ = line.split("\t")
            val_class_dir = f"tiny-imagenet/tiny-imagenet-200/val/{cls}"
            os.makedirs(val_class_dir, exist_ok=True)
            shutil.copyfile(
                f"tiny-imagenet/tiny-imagenet-200/val/images/{fn}",
                f"{val_class_dir}/{fn}",
            )

    # Remove the now empty 'images' folder to clean up
    shutil.rmtree("tiny-imagenet/tiny-imagenet-200/val/images")


def _get_transform():
    """Return the transformation pipeline for dataset images."""
    return T.Compose(
        [
            T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _get_imagenet_datasets() -> Tuple[ImageFolder, ImageFolder]:
    """Load and return Tiny ImageNet datasets for core and validation."""
    transform = _get_transform()

    # Loading train and validation datasets using ImageFolder
    tiny_imagenet_train = ImageFolder(
        root="tiny-imagenet/tiny-imagenet-200/train", transform=transform
    )
    tiny_imagenet_val = ImageFolder(
        root="tiny-imagenet/tiny-imagenet-200/val", transform=transform
    )

    return tiny_imagenet_train, tiny_imagenet_val


def get_imagenet_dataloaders(batch_size: int):
    """Return data loaders, checking if dataset exists and has the correct format."""
    if not _is_dataset_valid():
        print(
            "Dataset not found or not in the correct format. Downloading and formatting dataset."
        )
        _download_and_extract()  # Download and extract dataset if necessary
        _adjust_validation_format()  # Reformat validation set if needed
    else:
        print(
            "Dataset already exists and is in the correct format. Skipping download and formatting."
        )

    # Get train and validation datasets
    train_dataset, val_dataset = _get_imagenet_datasets()

    # Create DataLoader for both train and validation datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader
