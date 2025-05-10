import os
import shutil
import random
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def create_dataset_splits(source_dir, output_dir, test_size=0.2, val_size=0.2, random_state=42, force=False):
    """
    Split the dataset into train, validation, and test sets.

    Args:
        source_dir: Directory containing the original dataset
        output_dir: Directory to save the split dataset
        test_size: Proportion of the dataset to include in the test split
        val_size: Proportion of the training data to include in the validation split
        random_state: Random seed for reproducibility
        force: Whether to overwrite existing directories
    """
    # Create output directories if they don't exist
    if os.path.exists(output_dir) and not force:
        raise ValueError(f"Output directory {output_dir} already exists. Use --force to overwrite.")

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    for directory in [train_dir, val_dir, test_dir]:
        if os.path.exists(directory) and force:
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    # Get all classes (animal folders)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in tqdm(classes, desc="Processing classes"):
        # Create class directories in train, val, and test
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        # Get all images for this class
        class_dir = os.path.join(source_dir, cls)
        images = [img for img in os.listdir(class_dir) if img.endswith('.jpg') or img.endswith('.png')]

        # Split into train+val and test
        train_val_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=random_state)

        # Further split train+val into train and val
        train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=val_size, random_state=random_state)

        # Copy images to their respective directories
        for img in train_imgs:
            shutil.copy(
                os.path.join(class_dir, img),
                os.path.join(train_dir, cls, img)
            )

        for img in val_imgs:
            shutil.copy(
                os.path.join(class_dir, img),
                os.path.join(val_dir, cls, img)
            )

        for img in test_imgs:
            shutil.copy(
                os.path.join(class_dir, img),
                os.path.join(test_dir, cls, img)
            )

    # Print statistics
    print_dataset_stats(train_dir, val_dir, test_dir)

    return train_dir, val_dir, test_dir


def print_dataset_stats(train_dir, val_dir, test_dir):
    """Print statistics about the dataset splits."""
    train_classes = os.listdir(train_dir)
    val_classes = os.listdir(val_dir)
    test_classes = os.listdir(test_dir)

    train_count = sum(len(os.listdir(os.path.join(train_dir, cls))) for cls in train_classes)
    val_count = sum(len(os.listdir(os.path.join(val_dir, cls))) for cls in val_classes)
    test_count = sum(len(os.listdir(os.path.join(test_dir, cls))) for cls in test_classes)

    total_count = train_count + val_count + test_count

    print("Dataset Statistics:")
    print(f"Total images: {total_count}")
    print(f"Training images: {train_count} ({train_count / total_count:.2%})")
    print(f"Validation images: {val_count} ({val_count / total_count:.2%})")
    print(f"Test images: {test_count} ({test_count / total_count:.2%})")
    print(f"Number of classes: {len(train_classes)}")

    # Plot class distribution
    class_counts = {}
    for cls in train_classes:
        train_cls_count = len(os.listdir(os.path.join(train_dir, cls)))
        val_cls_count = len(os.listdir(os.path.join(val_dir, cls)))
        test_cls_count = len(os.listdir(os.path.join(test_dir, cls)))
        class_counts[cls] = (train_cls_count, val_cls_count, test_cls_count)

    # Take top 10 classes for a clearer visualization
    sorted_classes = sorted(class_counts.items(), key=lambda x: sum(x[1]), reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(sorted_classes))
    width = 0.25

    ax.bar(x - width, [c[1][0] for c in sorted_classes], width, label='Train')
    ax.bar(x, [c[1][1] for c in sorted_classes], width, label='Validation')
    ax.bar(x + width, [c[1][2] for c in sorted_classes], width, label='Test')

    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution (Top 10 Classes)')
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in sorted_classes], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(train_dir), 'class_distribution.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset by splitting into train, validation, and test sets")

    parser.add_argument("--source", type=str, required=True,
                        help="Directory containing the original dataset")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save the processed dataset")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split")
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Proportion of the training data to include in the validation split")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing directories")

    args = parser.parse_args()

    try:
        train_dir, val_dir, test_dir = create_dataset_splits(
            args.source,
            args.output,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            force=args.force
        )

        print(f"\nDataset successfully prepared:")
        print(f"- Training data: {train_dir}")
        print(f"- Validation data: {val_dir}")
        print(f"- Test data: {test_dir}")

    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
