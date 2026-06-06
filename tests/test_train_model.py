import os
import pytest
from unittest.mock import patch, MagicMock
from torchvision import transforms
from train_model import get_transforms, create_model

def test_get_transforms():
    """Test that transformations are constructed correctly."""
    train_tf, val_tf = get_transforms(image_size=224)
    
    # Train transforms should have data augmentation (more steps)
    assert isinstance(train_tf, transforms.Compose)
    assert len(train_tf.transforms) > 3
    
    # Val transforms should have fewer steps (Resize, ToTensor, Normalize)
    assert isinstance(val_tf, transforms.Compose)
    assert len(val_tf.transforms) == 3

@patch('train_model.timm.create_model')
def test_create_model(mock_timm_create):
    """Test model creation wrapper to ensure right timm call."""
    mock_model = MagicMock()
    mock_timm_create.return_value = mock_model
    
    model = create_model(num_classes=5, model_name="vit_base_patch16_224", pretrained=True)
    
    mock_timm_create.assert_called_once_with(
        "vit_base_patch16_224", pretrained=True, num_classes=5
    )
    assert model == mock_model
    
    # Test without pretrained
    mock_timm_create.reset_mock()
    model = create_model(num_classes=10, model_name="resnet50", pretrained=False)
    mock_timm_create.assert_called_once_with(
        "resnet50", pretrained=False, num_classes=10
    )
