import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

import config


def apply_backdoor_pattern(img):
    """
        Applies a backdoor pattern to the input image by modifying certain pixels.

        Args:
            img (Tensor): The input image tensor to which the backdoor pattern will be applied.

        Returns:
            Tensor: The image tensor with the backdoor pattern applied.
    """
    # Set specific pixel locations to 0 to create a visible backdoor pattern
    img[:, 2, 2] = 0
    img[:, 3, 3] = 0
    img[:, 4, 4] = 0
    img[:, 4, 2] = 0
    img[:, 2, 4] = 0
    return img


def _split_dataset_by_clients(dataset, num_clients):
    """
        Splits a dataset into `num_clients` non-empty subsets while preserving all samples.

        Args:
            dataset (Dataset): The dataset to split.
            num_clients (int): Number of client subsets.

        Returns:
            list: A list of dataset subsets, one for each client.
    """
    total_size = len(dataset)
    base_size = total_size // num_clients
    remainder = total_size % num_clients
    split_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_clients)]
    return random_split(dataset, split_sizes)


def get_num_classes(data_name):
    """
        Returns the number of classes for a supported dataset.

        Args:
            data_name (str): Dataset name.

        Returns:
            int: Number of classes.
    """
    classes_by_dataset = {
        "cifar10": 10,
        "fashion-mnist": 10,
        "imagenet": 1000,
    }
    return classes_by_dataset[data_name]


def get_dataset(data_name="cifar10"):
    """
        Loads the CIFAR-10 dataset and splits the training data into `num_clients` partitions for federated learning.

        Returns:
            tuple: A tuple containing:
                - train_splits (list of Subset): A list of datasets for each client.
                - testdata (Dataset): The testing dataset.
    """
    if data_name == "cifar10":
        transform = transforms.ToTensor()
        traindata = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        testdata = datasets.CIFAR10('./data', train=False, transform=transform)
    elif data_name == "fashion-mnist":
        # Convert grayscale images to 3 channels so the VGG input shape remains unchanged.
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        traindata = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        testdata = datasets.FashionMNIST('./data', train=False, transform=transform)
    elif data_name == "imagenet":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        train_dir = Path('./data/imagenet/train')
        val_dir = Path('./data/imagenet/val')
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                "ImageNet folders not found. Expected './data/imagenet/train' and './data/imagenet/val'."
            )
        traindata = datasets.ImageFolder(train_dir, transform=transform)
        testdata = datasets.ImageFolder(val_dir, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")

    # Split training data into subsets, one for each client
    train_splits = _split_dataset_by_clients(traindata, config.num_clients)

    return train_splits, testdata


def create_poisoned_data(dataset, target_class, num_poison=None):
    """
        Creates a dataset of poisoned images where a backdoor pattern is applied and all labels are set to the target class.

        Args:
            dataset (Dataset): The input dataset to poison.
            target_class (int): The class label assigned to the poisoned images.
            num_poison (int, optional): The number of poisoned samples to create. If None, poisons the entire dataset.

        Returns:
            TensorDataset: A dataset containing poisoned images and their corresponding target labels.
    """
    backdoor_images = []
    count = 0

    # Apply backdoor pattern to the dataset and set labels to the target class
    for img, _ in dataset:
        img = apply_backdoor_pattern(img)
        backdoor_images.append(img)
        count += 1
        if num_poison is not None and count == num_poison:
            break

    # Stack images into a single tensor and create corresponding labels
    backdoor_images = torch.stack(backdoor_images)
    backdoor_labels = torch.full((len(backdoor_images),), target_class, dtype=torch.int64)

    return TensorDataset(backdoor_images, backdoor_labels)


def get_poison_data(train_splits, testdata):
    """
        Creates loaders for benign and poisoned data, combining backdoor poisoned samples with benign samples for training.

        Args:
            train_splits (list of Subset): A list of datasets for each client.
            testdata (Dataset): The test dataset.

        Returns:
            tuple: Contains:
                - benign_loader (DataLoader): DataLoader for benign training data from the attacker client.
                - mixed_loader (DataLoader): DataLoader for a mix of benign and poisoned data.
                - poison_train_loader (DataLoader): DataLoader for training data with only poisoned samples.
                - poison_test_loader (DataLoader): DataLoader for testing data with only poisoned samples.
    """
    # Select the attacker's dataset from the client partitions
    attacker_dataset = train_splits[config.attacker_id]

    # Create a dataset with poisoned images (for the attacker)
    poison_train_dataset = create_poisoned_data(attacker_dataset, config.target_class, num_poison=config.num_poison)

    # Mix benign images with poisoned images for training
    mixed_images, mixed_labels = [], []
    for img, label in attacker_dataset:
        mixed_images.append(img)
        mixed_labels.append(torch.tensor(label))

    # Add poisoned samples to the mixed dataset
    for img, label in poison_train_dataset:
        mixed_images.append(img)
        mixed_labels.append(label)

    mixed_dataset = TensorDataset(torch.stack(mixed_images), torch.stack(mixed_labels))

    # Create poisoned test dataset
    poison_test_dataset = create_poisoned_data(testdata, config.target_class)

    # Create DataLoaders for different types of datasets
    benign_loader = DataLoader(attacker_dataset, batch_size=config.batch_size, shuffle=False)
    mixed_loader = DataLoader(mixed_dataset, batch_size=config.batch_size, shuffle=True)
    poison_train_loader = DataLoader(poison_train_dataset, batch_size=config.batch_size, shuffle=False)
    poison_test_loader = DataLoader(poison_test_dataset, batch_size=config.batch_size, shuffle=True)

    return benign_loader, mixed_loader, poison_train_loader, poison_test_loader
