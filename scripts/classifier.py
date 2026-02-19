"""
scripts/classifier.py
CIFAR-10 image classifier using pretrained ResNet (transfer learning).

Train:
  python scripts/classifier.py --train                     # ResNet-50, 25 epochs (recommended)
  python scripts/classifier.py --train --arch resnet18     # fast / lightweight
  python scripts/classifier.py --train --epochs 30 --arch resnet50
"""
import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Input size expected by ResNet
INPUT_SIZE = 224


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 10, device: str = 'cpu', arch: str = 'resnet50'):
    """
    Build a pretrained ResNet model with the final FC layer replaced for CIFAR-10.

    Args:
        num_classes: Number of output classes.
        device: 'cpu' or 'cuda'.
        arch: 'resnet18' or 'resnet50'. ResNet-50 gives higher accuracy.
    """
    try:
        if arch == 'resnet50':
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except ImportError:
        # Older torchvision fallback
        model = getattr(models, arch)(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_image(weights_path: str, image_tensor: torch.Tensor,
                  device: str = 'cpu', arch: str = 'resnet50') -> str:
    """
    Predict the CIFAR-10 class of a single image using Test-Time Augmentation (TTA).

    TTA averages predictions over 5 different crops/flips of the input image,
    which typically improves accuracy by ~0.5-1% compared to a single forward pass.

    Args:
        weights_path: Path to saved model weights (.pth).
        image_tensor: Pre-transformed image tensor of shape (C, H, W).
        device: 'cpu' or 'cuda'.
        arch: Architecture used during training ('resnet18' or 'resnet50').

    Returns:
        str: Predicted class label with confidence.
    """
    model = build_model(num_classes=10, device=device, arch=arch)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
    image_tensor = image_tensor.to(device)

    # Test-Time Augmentation: 5 augmented views → averaged logits
    tta_transforms = [
        transforms.Compose([]),                                          # original
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),   # h-flip
        transforms.Compose([transforms.RandomCrop(INPUT_SIZE, padding=16)]),   # crop
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                            transforms.RandomCrop(INPUT_SIZE, padding=16)]),   # flip+crop
        transforms.Compose([transforms.CenterCrop(int(INPUT_SIZE * 0.9)),
                            transforms.Resize((INPUT_SIZE, INPUT_SIZE))]),      # center-crop
    ]

    logits_sum = None
    with torch.no_grad():
        for t in tta_transforms:
            aug = t(image_tensor)
            out = model(aug)
            logits_sum = out if logits_sum is None else logits_sum + out

    probabilities = torch.softmax(logits_sum, dim=1)[0]
    pred = probabilities.argmax().item()
    confidence = probabilities[pred].item() * 100
    return f"{CLASSES[pred]} ({confidence:.1f}% confidence)"


# ── Training ───────────────────────────────────────────────────────────────────
def train(epochs: int = 25, save_path: str = 'models/cnn_classifier.pth',
          arch: str = 'resnet50'):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Training {arch.upper()} on {device} for {epochs} epochs")

    # ── Augmentation ──────────────────────────────────────────────────────────
    # AutoAugment (CIFAR-10 policy) + RandomErasing for strong regularisation
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Data ──────────────────────────────────────────────────────────────────
    batch_size = 128
    train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                             download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                             download=True, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device == 'cuda'))
    test_loader  = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=(device == 'cuda'))

    model = build_model(num_classes=10, device=device, arch=arch)

    # ── Phase 1: Warm up the new head (freeze backbone) ──────────────────────
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    # Label smoothing reduces overconfidence and acts as regularisation
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    head_epochs = max(3, epochs // 5)
    optimizer_head = optim.Adam(model.fc.parameters(), lr=1e-3)
    print(f"\n--- Phase 1: Training head only ({head_epochs} epochs) ---")
    _run_epochs(model, train_loader, test_loader, criterion,
                optimizer_head, head_epochs, device)

    # ── Phase 2: Fine-tune all layers with OneCycleLR ─────────────────────────
    for param in model.parameters():
        param.requires_grad = True

    remaining = epochs - head_epochs
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n],
         'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-4)

    # OneCycleLR: ramps up LR then anneals → faster, often better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-3, 5e-3],
        steps_per_epoch=len(train_loader),
        epochs=remaining,
        pct_start=0.3,
        anneal_strategy='cos',
    )

    print(f"\n--- Phase 2: Fine-tuning all layers ({remaining} epochs) ---")
    best_wts = _run_epochs(model, train_loader, test_loader, criterion,
                           optimizer, remaining, device,
                           scheduler=scheduler, epoch_offset=head_epochs,
                           return_best=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_wts, save_path)
    print(f"\n✅ Best model saved to {save_path}")


# ── Epoch runner ───────────────────────────────────────────────────────────────
def _run_epochs(model, train_loader, test_loader, criterion, optimizer, epochs,
                device, scheduler=None, epoch_offset=0, return_best=False):
    """Run training epochs, optionally returning the best model weights by val-acc."""
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    step_scheduler = isinstance(scheduler,
                                optim.lr_scheduler.OneCycleLR)

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step_scheduler:          # OneCycleLR steps per batch
                scheduler.step()

        if scheduler and not step_scheduler:   # epoch-level schedulers
            scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + epoch_offset:>3}/{epoch_offset + epochs}  "
              f"loss={running_loss / len(train_loader):.3f}  "
              f"val_acc={val_acc:.4f}  lr={current_lr:.2e}")

        # Track best model weights
        if return_best and val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            print(f"           ✔ New best val_acc={best_acc:.4f}")

    if return_best:
        model.load_state_dict(best_wts)   # restore best for caller
        print(f"\nBest validation accuracy achieved: {best_acc:.4f} "
              f"({best_acc * 100:.2f}%)")
        return best_wts
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Eval CIFAR-10 classifier")
    parser.add_argument('--train',  action='store_true',
                        help="Run training")
    parser.add_argument('--epochs', type=int, default=25,
                        help="Total training epochs (default: 25)")
    parser.add_argument('--save',   type=str, default='models/cnn_classifier.pth',
                        help="Path to save model weights")
    parser.add_argument('--arch',   type=str, default='resnet50',
                        choices=['resnet18', 'resnet50'],
                        help="Backbone architecture (default: resnet50)")
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, save_path=args.save, arch=args.arch)
    else:
        print("Use --train to start training.\n"
              "Examples:\n"
              "  python scripts/classifier.py --train                    "
              "# ResNet-50, 25 epochs\n"
              "  python scripts/classifier.py --train --arch resnet18    "
              "# lighter / faster\n"
              "  python scripts/classifier.py --train --epochs 30 --arch resnet50")
