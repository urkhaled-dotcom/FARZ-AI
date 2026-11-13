"""MobileNetV3-Small model factory."""
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from ml.src.data.dataset import CLASS_NAMES

def build_model(num_classes=len(CLASS_NAMES), pretrained=True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
