"""Dataset loading with Dirichlet-based Non-IID partitioning."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from .config import BATCH_SIZE, DATA_ROOT

# ── Module-level cache (avoids repeated downloads/loads) ──────────
_dataset_cache = {}


# ── utility: Dirichlet partition ─────────────────────────────────
def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 42):
    """
    Split dataset indices into `num_clients` subsets according to a Dirichlet
    distribution with concentration `alpha`.

    Returns
    -------
    client_indices : list of list of int
        client_indices[c] holds the global indices assigned to client c.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == k)[0] for k in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = class_indices[k].copy()
        rng.shuffle(idx_k)
        # sample proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # ensure every client gets at least a minimal share (especially for small alpha)
        proportions = np.maximum(proportions, 1e-6)
        proportions /= proportions.sum()

        cumsum = np.cumsum(proportions)
        cumsum[-1] = 1.0  # fix rounding
        boundaries = (cumsum * len(idx_k)).astype(int)
        start = 0
        for c in range(num_clients):
            end = boundaries[c]
            client_indices[c].extend(idx_k[start:end].tolist())
            start = end

    for c in range(num_clients):
        client_indices[c] = rng.permutation(client_indices[c]).tolist()

    return client_indices


def build_dataloaders_for_client(
    dataset: Dataset,
    client_idx: list,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
):
    """Create a DataLoader for a single client given its index list."""
    sub = Subset(dataset, client_idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── Dataset factory ──────────────────────────────────────────────
def get_raw_datasets(dataset_name: str):
    """Return (train_dataset, test_dataset) torchvision objects.  Uses cache."""
    if dataset_name in _dataset_cache:
        return _dataset_cache[dataset_name]

    if dataset_name == "FashionMNIST":
        mean, std = [0.2861], [0.3530]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        train = datasets.FashionMNIST(DATA_ROOT, train=True,  download=True, transform=transform)
        test  = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
        channel, im_size, num_classes = 1, (28, 28), 10

    elif dataset_name == "CIFAR10":
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        train = datasets.CIFAR10(DATA_ROOT, train=True,  download=True, transform=transform)
        test  = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
        channel, im_size, num_classes = 3, (32, 32), 10

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    result = (train, test, channel, im_size, num_classes)
    _dataset_cache[dataset_name] = result
    return result


def get_client_loaders(dataset_name: str, alpha: float, num_clients: int, seed: int = 42):
    """
    Return:
        train_loaders : list[DataLoader] – one per client
        test_loader   : DataLoader for the global test set
        client_indices : list[list[int]]
        channel, im_size, num_classes
    """
    train_ds, test_ds, channel, im_size, num_classes = get_raw_datasets(dataset_name)
    labels = np.array([train_ds.targets[i].item() if isinstance(train_ds.targets[i], torch.Tensor)
                        else train_ds.targets[i] for i in range(len(train_ds))])
    client_indices = dirichlet_partition(labels, num_clients, alpha, seed)

    train_loaders = [
        build_dataloaders_for_client(train_ds, idx) for idx in client_indices
    ]
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loaders, test_loader, client_indices, channel, im_size, num_classes


# ── Convenience wrapper (used by server.py) ─────────────────────
def get_dataloaders(dataset: str, num_clients: int, alpha: float,
                    batch_size: int = BATCH_SIZE, seed: int = 42):
    """
    Return:
        user_loaders   : list[DataLoader] – one per client (training)
        user_datasets  : list[TensorDataset] – raw TensorDataset per client
        test_loader    : DataLoader for the global test set
    """
    train_loaders, test_loader, client_indices, channel, im_size, num_classes = \
        get_client_loaders(dataset, alpha, num_clients, seed)

    # Also build raw TensorDatasets for each client (needed by FedSGA etc.)
    # ── disk cache for all_imgs/all_labs (avoids restacking every run) ──
    cache_dir = os.path.join(DATA_ROOT, "stacked_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{dataset}_tensors.pt")

    if os.path.exists(cache_file):
        print(f"  Loading cached tensors from {cache_file} ...", flush=True)
        data = torch.load(cache_file, map_location="cpu")
        all_imgs = data["imgs"]
        all_labs = data["labs"]
    else:
        train_ds, _, _, _, _ = get_raw_datasets(dataset)
        n = len(train_ds)
        all_imgs_list, all_labs_list = [], []
        tmp_loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE * 4, 1024), shuffle=False, num_workers=0)
        n_loaded = 0
        print(f"  Stacking {n} training images into tensors...", flush=True)
        for imgs, labs in tmp_loader:
            all_imgs_list.append(imgs)
            all_labs_list.append(labs)
            n_loaded += imgs.size(0)
            if n_loaded % 5000 <= len(imgs):
                print(f"    {n_loaded}/{n} ({n_loaded/n*100:.1f}%)", flush=True)
        all_imgs = torch.cat(all_imgs_list)
        all_labs = torch.cat(all_labs_list).long()
        torch.save({"imgs": all_imgs, "labs": all_labs}, cache_file)
        print(f"  Saved cached tensors to {cache_file}", flush=True)

    user_datasets = []
    for idx_list in client_indices:
        user_datasets.append(TensorDataset(all_imgs[idx_list], all_labs[idx_list]))

    return train_loaders, user_datasets, test_loader


# ── FedSGA helper: extract single-class subset ──────────────────
def split_by_class(dataset: TensorDataset, target_class: int,
                   num_classes: int = 10) -> TensorDataset:
    """Return a TensorDataset containing only samples of *target_class*."""
    imgs_list, labs_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y.item() == target_class:
            imgs_list.append(x.unsqueeze(0))
            labs_list.append(y)
    if not imgs_list:
        return None
    return TensorDataset(torch.cat(imgs_list),
                          torch.tensor(labs_list, dtype=torch.long))