"""Unified client implementations for federated unlearning.

Supports 5 algorithms:
  - fedsga       : Gradient Ascent on unlearn-set (Baseline 1)
  - neurotoxin   : Neurotoxin backdoor unlearn (Baseline 2)
  - quickdrop    : FedQuickDrop with data distillation (Baseline 3)
  - ddfu_client  : Proposed 1 – same Neurotoxin formula, all local data
  - ddfu_sample  : Proposed 2 – half neurotoxin g1 + half FedAvg g2
"""

import copy
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    UNLEARN_TARGET_CLASS, GAMMA, BETA, DEVICE,
    LOCAL_EPOCHS, BATCH_SIZE,
)


# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader) -> tuple:
    """Return (accuracy, average_loss)."""
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data)
            running_loss += criterion(out, target).item() * data.size(0)
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return correct / max(total, 1), running_loss / max(total, 1)


def split_dataset_half(dataset: TensorDataset) -> tuple:
    """Split dataset into two halves."""
    n = len(dataset)
    n1 = n // 2
    n2 = n - n1
    imgs1, labs1 = [], []
    imgs2, labs2 = [], []
    for i in range(n):
        x, y = dataset[i]
        if i < n1:
            imgs1.append(x.unsqueeze(0))
            labs1.append(y)
        else:
            imgs2.append(x.unsqueeze(0))
            labs2.append(y)
    d1 = TensorDataset(torch.cat(imgs1), torch.tensor(labs1, dtype=torch.long))
    d2 = TensorDataset(torch.cat(imgs2), torch.tensor(labs2, dtype=torch.long))
    return d1, d2


# ────────────────────────────────────────────────────────────────
#  Baseline 1 – FedSGA (Gradient Ascent on unlearnset)
# ────────────────────────────────────────────────────────────────

def fedsga_unlearn(model: nn.Module, unlearn_loader: DataLoader,
                   epochs: int = LOCAL_EPOCHS, lr: float = 0.01) -> nn.Module:
    """Gradient Ascent on the unlearn-set.

    SGA = save copy → train (SGD minimize) → final = orig + (orig - trained)
    This effectively reverses the gradient direction → gradient ascent.
    """
    orig = copy.deepcopy(model)
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in unlearn_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    # SGA step: reverse the update
    orig_sd = orig.state_dict()
    curr_sd = model.state_dict()
    final_sd = {}
    for k in orig_sd:
        final_sd[k] = orig_sd[k] + (orig_sd[k] - curr_sd[k])
    model.load_state_dict(final_sd)
    return model


# ────────────────────────────────────────────────────────────────
#  Baseline 2 & Proposed 1 – Neurotoxin Unlearn
# ────────────────────────────────────────────────────────────────

def _build_backdoor_loader(data_loader: DataLoader, target_class: int):
    """Replace all labels in *data_loader* with target_class.

    Returns a new DataLoader whose targets are all target_class.
    """
    dataset = data_loader.dataset
    images = []
    for i in range(len(dataset)):
        img, _ = dataset[i]
        images.append(img.unsqueeze(0))
    images = torch.cat(images).to(DEVICE)
    labels = torch.full((len(dataset),), target_class, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels),
                       batch_size=data_loader.batch_size, shuffle=True)


def neurotoxin_unlearn(model: nn.Module,
                       train_loader: DataLoader,
                       backdoor_loader: DataLoader,
                       epochs: int = LOCAL_EPOCHS,
                       lr: float = 0.01) -> nn.Module:
    """
    Neurotoxin unlearning::

        clean_loss    = Σ CE(model(x), y)           on train_loader
        backdoor_loss = - Σ CE(model(x), target)    on backdoor_loader
        penalty       = Σ ||clean_grad_i / backdoor_grad_i · (p_i − p_i⁰)||₁
        total_loss    = clean_loss + γ * backdoor_loss + β * penalty
    """
    crit = nn.CrossEntropyLoss()
    # save original parameters for penalty
    orig_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    model.train()

    for _ in range(epochs):
        # ‣ collect all batches once per epoch (simplest approach)
        train_batches = list(train_loader)
        backdoor_batches = list(backdoor_loader)

        clean_loss_sum = 0.0
        backdoor_loss_sum = 0.0

        # ---- forward losses (detached) ----
        for x, y in train_batches:
            x, y = x.to(DEVICE), y.to(DEVICE)
            clean_loss_sum += crit(model(x), y)

        for x, y in backdoor_batches:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # backdoor_loss is negative CE
            backdoor_loss_sum -= crit(model(x), y)

        total_loss = clean_loss_sum + GAMMA * backdoor_loss_sum

        # ---- clean importance gradients (retain graph for penalty) ----
        model.zero_grad()
        clean_loss_sum.backward(retain_graph=True)
        clean_grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]

        # ---- backdoor importance gradients ----
        model.zero_grad()
        backdoor_loss_sum.backward(retain_graph=True)
        bd_grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]

        # ---- penalty L2 (L1-norm version from original) ----
        cur_params = [p for p in model.parameters() if p.requires_grad]
        penalty = 0.0
        for i, p in enumerate(cur_params):
            importance = torch.nan_to_num(
                torch.div(clean_grads[i], bd_grads[i]), nan=0.0, posinf=0.0, neginf=0.0
            )
            penalty += torch.norm(importance * torch.abs(p - orig_params[i]), 1)

        # ---- final loss = clean + γ·backdoor + β·penalty ----
        unlearn_loss = total_loss + BETA * penalty
        model.zero_grad()
        unlearn_loss.backward()
        # manual SGD step
        with torch.no_grad():
            for p in cur_params:
                p -= lr * p.grad

    return model


# ────────────────────────────────────────────────────────────────
#  Baseline 3 – FedQuickDrop (DC / DSA synthetic distillation)
# ────────────────────────────────────────────────────────────────

# Simple differentiable augmentation (DSA-lite)
def _diff_augment(imgs: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Apply random crop + horizontal flip (CIFAR-style)."""
    bs, c, h, w = imgs.shape
    g = torch.Generator(device=imgs.device)
    g.manual_seed(seed)
    # random crop (pad 4 then crop back)
    pad = nn.ReflectionPad2d(4)
    aug = pad(imgs)
    # random offset 0..8
    offset_y = torch.randint(0, 8, (bs,), generator=g, device=imgs.device)
    offset_x = torch.randint(0, 8, (bs,), generator=g, device=imgs.device)
    cropped = torch.stack([
        aug[i, :, oy:oy + h, ox:ox + w]
        for i, (oy, ox) in enumerate(zip(offset_y, offset_x))
    ])
    # random flip
    flip = torch.randint(0, 2, (bs,), generator=g, device=imgs.device).float()
    flip_idx = flip > 0.5
    if flip_idx.any():
        cropped[flip_idx] = torch.flip(cropped[flip_idx], dims=[3])
    return cropped


def quickdrop_distill_and_train(
    model: nn.Module,
    train_loader: DataLoader,
    num_classes: int = 10,
    distill_iters: int = 100,    # outer loop iters for DC
    lr_img: float = 0.1,
    lr_net: float = 0.01,
    ipc_ratio: float = 0.05,     # images-per-class ratio
    use_dsa: bool = True,
) -> nn.Module:
    """Dataset Condensation / DSA distillation → train on synthetic data.
    Simplified version adapted from FedQuickDrop.
    """
    dataset = train_loader.dataset
    im_size = dataset[0][0].shape  # (C, H, W)
    c, h, w = im_size

    # ── group by class ──
    indices_class = [[] for _ in range(num_classes)]
    for i in range(len(dataset)):
        indices_class[dataset[i][1]].append(i)
    all_imgs = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(DEVICE)
    all_labs = torch.tensor([dataset[i][1] for i in range(len(dataset))],
                            dtype=torch.long, device=DEVICE)

    def get_images(class_id, n):
        idx = np.random.permutation(indices_class[class_id])[:n]
        return all_imgs[idx]

    # ── ipc per class ──
    ipcs = [max(1, math.ceil(len(indices_class[c]) * ipc_ratio)) if indices_class[c] else 0
            for c in range(num_classes)]
    total_syn = sum(ipcs)

    # ── init synth images ──
    syn_imgs = torch.randn(total_syn, c, h, w, device=DEVICE, requires_grad=True)
    for cl in range(num_classes):
        if ipcs[cl]:
            syn_imgs.data[sum(ipcs[:cl]):sum(ipcs[:cl + 1])] = get_images(cl, ipcs[cl]).detach()
    syn_labs = torch.cat([
        torch.full((ipcs[c],), c, dtype=torch.long, device=DEVICE)
        for c in range(num_classes) if ipcs[c]
    ])

    # ── optimisers ──
    opt_img = optim.SGD([syn_imgs], lr=lr_img, momentum=0.5)

    # ── distillation loop ──
    crit = nn.CrossEntropyLoss()
    for it in range(distill_iters):
        # random re-init model for each iteration (standard DC practice)
        # BUT this is too slow; we use the current model as the "net" to match gradients
        net = copy.deepcopy(model)
        net.train()
        net_params = list(net.parameters())
        opt_net = optim.SGD(net.parameters(), lr=lr_net)

        # freeze BN stats
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for ol in range(1):  # outer loop
            loss_total = torch.tensor(0.0, device=DEVICE)
            for cl in range(num_classes):
                if ipcs[cl] <= 1 and len(indices_class[cl]) <= 1:
                    continue
                n_real = min(256, len(indices_class[cl]))
                imgs_real = get_images(cl, n_real)
                labs_real = torch.full((n_real,), cl, dtype=torch.long, device=DEVICE)
                imgs_syn = syn_imgs[sum(ipcs[:cl]):sum(ipcs[:cl + 1])]
                labs_syn = torch.full((ipcs[cl],), cl, dtype=torch.long, device=DEVICE)

                if use_dsa:
                    seed = int(time.time() * 1000) % 100000
                    imgs_real_aug = _diff_augment(imgs_real, seed)
                    imgs_syn_aug = _diff_augment(imgs_syn, seed)
                else:
                    imgs_real_aug = imgs_real
                    imgs_syn_aug = imgs_syn

                # gradients from real
                net.zero_grad()
                out_real = net(imgs_real_aug)
                loss_real = crit(out_real, labs_real)
                gw_real = torch.autograd.grad(loss_real, net_params, create_graph=False)
                gw_real = [g.detach().clone() for g in gw_real]

                # gradients from synth
                net.zero_grad()
                out_syn = net(imgs_syn_aug)
                loss_syn = crit(out_syn, labs_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_params, create_graph=True)

                # gradient matching loss
                loss_match = 0.0
                for gr, gs in zip(gw_real, gw_syn):
                    loss_match += nn.functional.mse_loss(gs, gr)
                loss_total += loss_match

            opt_img.zero_grad()
            loss_total.backward()
            opt_img.step()

            # train net on synth for a few steps
            if ol == 0:
                syn_ds = TensorDataset(syn_imgs.detach(), syn_labs.detach())
                syn_loader = DataLoader(syn_ds, batch_size=min(256, total_syn), shuffle=True)
                for _ in range(5):  # inner loop
                    for sx, sy in syn_loader:
                        sx, sy = sx.to(DEVICE), sy.to(DEVICE)
                        opt_net.zero_grad()
                        l = crit(net(sx), sy)
                        l.backward()
                        opt_net.step()
                model.load_state_dict(net.state_dict())  # update model with distilled

    # ── final training on synthetic data ──
    final_syn = TensorDataset(syn_imgs.detach(), syn_labs.detach())
    final_loader = DataLoader(final_syn, batch_size=BATCH_SIZE, shuffle=True)
    opt_final = optim.SGD(model.parameters(), lr=lr_net, momentum=0.5)
    model.train()
    for _ in range(min(10, distill_iters)):
        for x, y in final_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt_final.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt_final.step()

    return model


# ────────────────────────────────────────────────────────────────
#  Proposed 2 – ddfu-sample: half neurotoxin + half FedAvg
# ────────────────────────────────────────────────────────────────

def ddfu_sample_update(model: nn.Module,
                       train_loader: DataLoader,
                       backdoor_loader: DataLoader,
                       epochs: int = LOCAL_EPOCHS,
                       lr: float = 0.01) -> nn.Module:
    """Split data in half: first half → neurotoxin gradient g₁,
    second half → standard FedAvg gradient g₂, then g = (g₁+g₂)/2."""
    dataset = train_loader.dataset
    d1, d2 = split_dataset_half(dataset)

    # Build loaders for each half
    loader1 = DataLoader(d1, batch_size=train_loader.batch_size, shuffle=True)
    loader2 = DataLoader(d2, batch_size=train_loader.batch_size, shuffle=True)

    # ---- Neurotoxin on first half ----
    crit = nn.CrossEntropyLoss()
    orig_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    # (a) neurotoxin gradients g₁
    model.train()
    for _ in range(epochs):
        b1 = list(loader1)
        bdb = list(backdoor_loader)

        clean_sum = sum(crit(model(x.to(DEVICE)), y.to(DEVICE)) for x, y in b1)
        bd_sum = -sum(crit(model(x.to(DEVICE)), y.to(DEVICE)) for x, y in bdb)

        model.zero_grad()
        clean_sum.backward(retain_graph=True)
        cg = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]

        model.zero_grad()
        bd_sum.backward(retain_graph=True)
        bg = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]

        cur_p = [p for p in model.parameters() if p.requires_grad]
        penalty = 0.0
        for i, p in enumerate(cur_p):
            imp = torch.nan_to_num(torch.div(cg[i], bg[i]), nan=0.0, posinf=0.0, neginf=0.0)
            penalty += torch.norm(imp * torch.abs(p - orig_params[i]), 1)

        loss = clean_sum + GAMMA * bd_sum + BETA * penalty
        model.zero_grad()
        loss.backward()
        g1 = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]

    # (b) FedAvg gradients g₂ on second half
    model2 = copy.deepcopy(model)  # start from same model
    model2.train()
    opt2 = optim.SGD(model2.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader2:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt2.zero_grad()
            loss = crit(model2(x), y)
            loss.backward()
            opt2.step()

    # g₂ = original_params - updated_params (i.e., the negative of the accumulated update)
    g2 = []
    for p_orig, p_new in zip(model.parameters(), model2.parameters()):
        if p_orig.requires_grad:
            g2.append((p_orig.detach() - p_new.detach()).clone())

    # Merge: g = (g₁ + g₂) / 2
    with torch.no_grad():
        idx = 0
        for p in model.parameters():
            if p.requires_grad:
                p -= lr * (g1[idx] + g2[idx]) / 2.0
                idx += 1

    return model


# ────────────────────────────────────────────────────────────────
#  Unified Client Runner
# ────────────────────────────────────────────────────────────────

def run_client(mode: str, model: nn.Module, train_loader: DataLoader,
               unlearn_loader: DataLoader = None, num_classes: int = 10,
               epochs: int = LOCAL_EPOCHS, lr: float = 0.01,
               use_dsa: bool = True) -> nn.Module:
    """Dispatch to the correct unlearning algorithm.

    Parameters
    ----------
    mode : str
        One of 'fedsga', 'neurotoxin', 'quickdrop', 'ddfu_client', 'ddfu_sample'.
    model : nn.Module
        The global model (copied locally).
    train_loader : DataLoader
        The forget-client's full data loader.
    unlearn_loader : DataLoader, optional
        Only needed for 'fedsga' (single-class unlearnset).
    """
    model_copy = copy.deepcopy(model).to(DEVICE)

    if mode == "fedsga":
        assert unlearn_loader is not None
        return fedsga_unlearn(model_copy, unlearn_loader, epochs=epochs, lr=lr)

    elif mode == "neurotoxin":
        bd_loader = _build_backdoor_loader(train_loader, UNLEARN_TARGET_CLASS)
        return neurotoxin_unlearn(model_copy, train_loader, bd_loader,
                                  epochs=epochs, lr=lr)

    elif mode == "quickdrop":
        return quickdrop_distill_and_train(model_copy, train_loader,
                                           num_classes=num_classes, use_dsa=use_dsa,
                                           distill_iters=50, lr_img=0.1, lr_net=0.01)

    elif mode == "ddfu_client":
        bd_loader = _build_backdoor_loader(train_loader, UNLEARN_TARGET_CLASS)
        return neurotoxin_unlearn(model_copy, train_loader, bd_loader,
                                  epochs=epochs, lr=lr)

    elif mode == "ddfu_sample":
        bd_loader = _build_backdoor_loader(train_loader, UNLEARN_TARGET_CLASS)
        return ddfu_sample_update(model_copy, train_loader, bd_loader,
                                  epochs=epochs, lr=lr)

    else:
        raise ValueError(f"Unknown client mode: {mode}")