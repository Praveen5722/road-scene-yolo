import pytest
import torch
from medical_cnn.model import RegularizedCNN

def test_forward_shape_for_two_classes():
    model=RegularizedCNN(2); model.eval()
    with torch.no_grad(): output=model(torch.randn(2,3,224,224))
    assert output.shape == (2,2)

def test_rejects_single_class():
    with pytest.raises(ValueError): RegularizedCNN(1)

def test_dropout_validation():
    with pytest.raises(ValueError): RegularizedCNN(3,1.0)
