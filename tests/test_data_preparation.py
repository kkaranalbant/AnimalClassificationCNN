import os
import tempfile
import pytest
import numpy as np
from data_preparation import create_dataset_splits

def test_create_dataset_splits():
    # Use a temporary directory to avoid writing to the actual filesystem
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = os.path.join(temp_dir, 'source')
        output_dir = os.path.join(temp_dir, 'output')
        
        # Create dummy source directory with dummy classes
        classes = ['dog', 'cat']
        os.makedirs(source_dir)
        for cls in classes:
            class_path = os.path.join(source_dir, cls)
            os.makedirs(class_path)
            # create 10 dummy images per class
            for i in range(10):
                with open(os.path.join(class_path, f'img_{i}.jpg'), 'w') as f:
                    f.write('dummy')
        
        # Call the function
        train_dir, val_dir, test_dir = create_dataset_splits(
            source_dir=source_dir,
            output_dir=output_dir,
            test_size=0.2,
            val_size=0.2,
            random_state=42,
            force=True
        )
        
        # Assert directories exist
        assert os.path.exists(train_dir)
        assert os.path.exists(val_dir)
        assert os.path.exists(test_dir)
        
        # Assert class subdirectories exist in each split
        for split_dir in [train_dir, val_dir, test_dir]:
            for cls in classes:
                assert os.path.exists(os.path.join(split_dir, cls))
        
        # For 10 images, test_size=0.2 => 2 images in test
        # Remaining 8, val_size=0.2 => 2 images in val (approx depending on rounding)
        # Check totals
        for cls in classes:
            train_count = len(os.listdir(os.path.join(train_dir, cls)))
            val_count = len(os.listdir(os.path.join(val_dir, cls)))
            test_count = len(os.listdir(os.path.join(test_dir, cls)))
            assert train_count + val_count + test_count == 10
            assert test_count > 0
            assert train_count > 0

def test_create_dataset_splits_existing_dir_no_force():
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = os.path.join(temp_dir, 'source')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(source_dir)
        os.makedirs(output_dir)  # pre-create output
        
        with pytest.raises(ValueError, match="Output directory .* already exists"):
            create_dataset_splits(
                source_dir=source_dir,
                output_dir=output_dir,
                test_size=0.2,
                val_size=0.2,
                random_state=42,
                force=False
            )
