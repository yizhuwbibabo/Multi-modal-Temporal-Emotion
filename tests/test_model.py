import torch

from mmted.model import MMTED


def test_mmted_forward_and_modality_weights():
    model = MMTED(input_channels=2, eye_features=6, num_classes=4, embed_dim=32)
    output = model(torch.randn(3, 2, 16), torch.randn(3, 16, 6))
    assert output["logits"].shape == (3, 4)
    assert torch.allclose(output["fusion_weights"].sum(1), torch.ones(3), atol=1e-6)
