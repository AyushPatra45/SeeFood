"""One logit: sigmoid(logit) is the hot_dog score. No sigmoid before BCEWithLogitsLoss."""
import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

CLASS_TO_IDX = {"not_hot_dog": 0, "hot_dog": 1}
WEIGHTS = MobileNet_V2_Weights.IMAGENET1K_V2


def build_model(pretrained: bool = False):
    model = mobilenet_v2(weights=WEIGHTS if pretrained else None)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 1))
    return model


def configure_stage(model, stage):
    """Freeze backbone first, then train its last four blocks; hold BatchNorm fixed."""
    if stage not in {"head", "finetune"}:
        raise ValueError("stage must be head or finetune")
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.classifier.parameters():
        param.requires_grad_(True)
    if stage == "finetune":
        for block in model.features[15:]:
            for param in block.parameters():
                param.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_to_idx") != CLASS_TO_IDX:
        raise ValueError("Checkpoint class mapping does not match this application")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
