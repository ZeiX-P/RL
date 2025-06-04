# pizza steak sushi dataset
import zipfile
import requests
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


__all__ = ["get_dataloaders"]


def download_data(url: str, file_name: str):
    """
    Download the dataset from the given URL and save it to the specified file.
    """
    print("Downloading dataset...")
    response = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {file_name}")


def unzip_data(file_name: str, extract_to: str):
    """
    Unzip the dataset to the specified directory.
    """
    print("Unzipping the dataset...")
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def get_transform():
    """
    Create a transform to resize and normalize images for core.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize all images to 224x224
            transforms.ToTensor(),  # Convert images to tensor
            transforms.Normalize(  # Normalize with ImageNet means and std
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def load_datasets(data_dir: str, transform):
    """
    Load the train and test datasets using the ImageFolder class.
    """
    train_dset = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    test_dset = datasets.ImageFolder(f"{data_dir}/test", transform=transform)

    return train_dset, test_dset


def create_dataloaders(train_dset, test_dset, batch_size=32):
    """
    Create DataLoader instances for core and testing datasets.
    """
    train_dataloader = DataLoader(
        train_dset, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_dataloader = DataLoader(test_dset, batch_size=batch_size, num_workers=2)

    return train_dataloader, test_dataloader


def get_class_names(train_dset):
    """
    Get the class names from the core dataset.
    """
    return train_dset.classes


def get_dataloaders():
    # URL and file path configurations
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    file_name = "pizza_steak_sushi.zip"
    data_dir = "data/"

    # Download, unzip, and prepare data
    download_data(url, file_name)
    unzip_data(file_name, "data")

    transform = get_transform()

    # Load datasets, class_names, dataloaders
    train_dset, test_dset = load_datasets(data_dir, transform)
    class_names = get_class_names(train_dset)
    train_dataloader, test_dataloader = create_dataloaders(train_dset, test_dset)

    return train_dataloader, test_dataloader, class_names
