import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import datetime
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_transforms(image_size=224):
    """Get transforms for training and validation/testing."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def load_data(data_dir, batch_size=32, image_size=224, num_workers=4):
    """Load datasets and create data loaders."""
    train_transform, val_transform = get_transforms(image_size)

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Save class names mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open('class_mapping.json', 'w') as f:
        json.dump(idx_to_class, f)

    return train_loader, val_loader, test_loader, class_to_idx

def create_model(num_classes, model_name="vit_base_patch16_224", pretrained=True):
    """Create a Vision Transformer model."""
    print(f"Creating model: {model_name}")
    if pretrained:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    return model

def train_epoch(model, data_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(data_loader, desc="Training")

    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on the given data loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    metrics = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_metrics(train_metrics, val_metrics, output_dir):
    """Plot training and validation metrics."""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(train_metrics[metric], label=f'Train {metric}')
        plt.plot(val_metrics[metric], label=f'Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs. Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_plot.png'))
        plt.close()

def save_model(model, optimizer, epoch, best_metric, class_to_idx, output_dir, filename='model.pth'):
    """Save the model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'class_to_idx': class_to_idx
    }
    torch.save(state, os.path.join(output_dir, filename))

def train_model(model, train_loader, val_loader, test_loader, class_to_idx,
                output_dir, num_epochs=30, learning_rate=0.0003, weight_decay=1e-4,
                patience=5):
    """Train the model."""
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience // 2)

    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    train_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    val_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_results, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

        val_loss = val_results['loss']
        val_acc = val_results['accuracy']

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Precision: {val_results['precision']:.4f}, "
              f"Recall: {val_results['recall']:.4f}, "
              f"F1: {val_results['f1']:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0
            save_model(model, optimizer, epoch, best_val_acc, class_to_idx, output_dir, 'best_model.pth')
            print(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break

        train_metrics['loss'].append(train_loss)
        train_metrics['accuracy'].append(train_acc)

        for key in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            val_metrics[key].append(val_results[key])

        save_model(model, optimizer, epoch, best_val_acc, class_to_idx, output_dir)

    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\nEvaluating best model on test set...")
    test_results, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

    print(f"Test results: Accuracy: {test_results['accuracy']:.4f}, "
          f"Precision: {test_results['precision']:.4f}, "
          f"Recall: {test_results['recall']:.4f}, "
          f"F1: {test_results['f1']:.4f}")

    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_results,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
    }

    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    plot_confusion_matrix(test_targets, test_preds, class_names, output_dir)
    plot_metrics(train_metrics, val_metrics, output_dir)

    return model, results

def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer Model")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to processed dataset directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--model', type=str, default="vit_base_patch16_224")
    args = parser.parse_args()

    output_dir = f"./model_outputs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    config = {
        'data_dir': args.data_dir,
        'output_dir': output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        'model_name': args.model,
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print("Loading data...")
    train_loader, val_loader, test_loader, class_to_idx = load_data(
        args.data_dir, batch_size=args.batch_size, image_size=args.image_size
    )

    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")
    model = create_model(num_classes, model_name=args.model)
    model = model.to(device)

    print("Training model...")
    model, results = train_model(
        model, train_loader, val_loader, test_loader, class_to_idx,
        output_dir, num_epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=1e-4
    )

    print(f"Training complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()