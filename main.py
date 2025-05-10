import os
import argparse
import subprocess
import sys
import time
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80 + "\n")


def print_step(step_num, total_steps, text):
    """Print a formatted step."""
    print(f"\n[Step {step_num}/{total_steps}] {text}")
    print("-" * 80)


def validate_paths(args):
    """Validate the input paths."""
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return False

    if not os.path.isdir(args.data_dir):
        print(f"Error: The specified path is not a directory: {args.data_dir}")
        return False

    if args.output_dir and os.path.exists(args.output_dir) and not args.force:
        print(f"Error: Output directory already exists: {args.output_dir}")
        print("Use --force to overwrite existing directory")
        return False

    return True


import os
import argparse
import subprocess


def prepare_dataset(source_dir, output_dir, force=False, test_size=0.2, val_size=0.2, random_state=42):
    print("\n[Step 1/3] Preparing dataset")
    print("-" * 80)

    cmd = [
        "python3", "data_preparation.py",
        "--source", source_dir,
        "--output", output_dir,
        "--test-size", str(test_size),
        "--val-size", str(val_size),
        "--random-state", str(random_state),
    ]

    if force:  # Add --force flag if needed
        cmd.append("--force")

    subprocess.run(cmd, check=True)

def train_model(data_dir, args):
    print("\n[Step 2/3] Training model")
    print("-" * 80)

    cmd = [
        "python3", "train_model.py",
        "--data-dir", data_dir,  # <-- Bu satır doğru data_dir'i iletiyor
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--model", args.model,
        "--image-size", str(args.image_size)
    ]
    subprocess.run(cmd, check=True)

    return os.path.join("models", f"{args.model}.pth")


def evaluate_model(model_path, data_dir, image_size):
    print("\n[Step 3/3] Evaluating model")
    print("-" * 80)

    cmd = [
        "python3", "src/evaluate_model.py",
        "--model-path", model_path,
        "--data-dir", data_dir,
        "--image-size", str(image_size)
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="MultiZoo Main Pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to raw data directory (e.g. /home/kaan/Desktop/train)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing output directories")
    args = parser.parse_args()

    print("=" * 80)
    print("==================== MultiZoo Animal Classification Project ====================")
    print("=" * 80)

    processed_data_dir = "src/processed_zoo_dataset"

    prepare_dataset(args.data_dir, processed_data_dir, force=args.force)
    model_path = train_model(processed_data_dir, args)
    evaluate_model(model_path, processed_data_dir)


# if __name__ == "__main__":
#     main()
#


# def prepare_dataset(args):
#     """Prepare the dataset by splitting it into train/val/test."""
#     print_step(1, 3, "Preparing dataset")
#
#     script_path = os.path.join(os.path.dirname(__file__), "data_preparation.py")
#     cmd = [
#         sys.executable, script_path,
#         "--source", args.data_dir,
#         "--output", args.output_dir,
#         "--test-size", str(args.test_size),
#         "--val-size", str(args.val_size),
#         "--random-state", str(args.random_state)
#     ]
#
#     if args.force:
#         cmd.append("--force")
#
#     print(f"Running: {' '.join(cmd)}")
#     subprocess.run(cmd, check=True)
#
#     return os.path.join(args.output_dir, "processed_zoo_dataset")


def train_model(processed_data_dir, args):
    """Train the model with the processed dataset."""
    print_step(2, 3, "Training model")

    script_path = os.path.join(os.path.dirname(__file__), "train_model.py")
    cmd = [
        sys.executable, script_path,
        "--data-dir", processed_data_dir,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--model", args.model,
        "--image-size", str(args.image_size)
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Find the most recent model output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dirs = [d for d in os.listdir(base_dir) if d.startswith('model_outputs_')]
    if not model_dirs:
        print("Error: No model output directories found after training")
        return None

    most_recent_dir = max(model_dirs)
    model_path = os.path.join(base_dir, most_recent_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    return model_path


def run_interface(model_path):
    """Run the graphical interface with the trained model."""
    print_step(3, 3, "Running interface")

    script_path = os.path.join(os.path.dirname(__file__), "animal_classifier_app.py")
    cmd = [
        sys.executable, script_path,
        model_path
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# def main():
#     """Main function to run the MultiZoo animal classification project."""
#     parser = argparse.ArgumentParser(description="MultiZoo Animal Classification Project")
#
#     # Data parameters
#     parser.add_argument("--data-dir", type=str, required=True,
#                         help="Directory containing the original dataset")
#     parser.add_argument("--output-dir", type=str, default="./processed_zoo_dataset",
#                         help="Directory to save the processed dataset")
#     parser.add_argument("--test-size", type=float, default=0.2,
#                         help="Proportion of the dataset to include in the test split")
#     parser.add_argument("--val-size", type=float, default=0.2,
#                         help="Proportion of the training data to include in the validation split")
#
#     # Model parameters
#     parser.add_argument("--batch-size", type=int, default=32,
#                         help="Batch size for training")
#     parser.add_argument("--epochs", type=int, default=30,
#                         help="Number of epochs to train")
#     parser.add_argument("--learning-rate", type=float, default=0.0003,
#                         help="Learning rate for the optimizer")
#     parser.add_argument("--model", type=str, default="vit_base_patch16_224",
#                         help="Transformer model to use")
#     parser.add_argument("--image-size", type=int, default=224,
#                         help="Size to resize images to")
#
#     # General parameters
#     parser.add_argument("--random-state", type=int, default=42,
#                         help="Random state for reproducibility")
#     parser.add_argument("--force", action="store_true",
#                         help="Force overwrite existing directories")
#     parser.add_argument("--skip-prepare", action="store_true",
#                         help="Skip dataset preparation step")
#     parser.add_argument("--skip-train", action="store_true",
#                         help="Skip model training step")
#     parser.add_argument("--skip-interface", action="store_true",
#                         help="Skip running the interface")
#
#     args = parser.parse_args()
#
#     # Print header
#     print_header("MultiZoo Animal Classification Project")
#
#     # Validate paths
#     if not validate_paths(args):
#         return
#
#     # Prepare dataset
#     processed_data_dir = args.output_dir
#     if not args.skip_prepare:
#         processed_data_dir = prepare_dataset(args)
#     else:
#         print("Skipping dataset preparation")
#
#     # Train model
#     model_path = None
#     if not args.skip_train:
#         model_path = train_model(processed_data_dir, args)
#         if not model_path:
#             print("Error training model. Exiting.")
#             return
#     else:
#         print("Skipping model training")
#         # Try to find the most recent model
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         model_dirs = [d for d in os.listdir(base_dir) if d.startswith('model_outputs_')]
#         if model_dirs:
#             most_recent_dir = max(model_dirs)
#             model_path = os.path.join(base_dir, most_recent_dir, 'best_model.pth')
#             if not os.path.exists(model_path):
#                 print(f"Error: Model file not found at {model_path}")
#                 if not args.skip_interface:
#                     print("Cannot run interface without a trained model. Exiting.")
#                     return
#
#     # Run interface
#     if not args.skip_interface and model_path:
#         run_interface(model_path)
#     elif not args.skip_interface:
#         print("Cannot run interface without a trained model")
#     else:
#         print("Skipping interface")
#
#     print_header("Process Complete")


if __name__ == "__main__":
    main()
