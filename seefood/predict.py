"""The same deterministic preprocessing powers CLI, API, and Grad-CAM."""
import argparse
import base64
import io
import json
import math
from pathlib import Path
import threading
import time
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.nn import functional as F
from torchvision import transforms

from seefood.data import image_transform
from seefood.model import load_checkpoint

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def decode_image(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Upload a non-empty image smaller than 8 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise ValueError("Only JPEG, PNG, and WebP images are supported.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 16 megapixel limit.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("The file is not a valid, readable image.") from exc


def decision(probability, threshold):
    positive = probability >= threshold
    bounded = min(max(probability, 1e-12), 1-1e-12)
    entropy = -(bounded*math.log(bounded)+(1-bounded)*math.log(1-bounded))
    reasons = []
    if abs(probability - threshold) < 0.10:
        reasons.append("near_decision_boundary")
    if entropy > 0.50:
        reasons.append("high_binary_entropy")
    return {"label": "Hotdog!" if positive else "Not Hotdog!", "class_name": "hot_dog" if positive else "not_hot_dog",
            "probability_hot_dog": probability, "confidence": probability if positive else 1 - probability,
            "threshold": threshold, "uncertain": bool(reasons), "binary_entropy_nats": entropy,
            "uncertainty_reasons": reasons}


class Predictor:
    def __init__(self, model_path="artifacts/model.pt", device="cpu"):
        torch.set_num_threads(4)
        self.device = torch.device(device)
        self.model, self.metadata = load_checkpoint(model_path, self.device)
        self.transform = image_transform()
        self.lock = threading.Lock()

    def predict_bytes(self, raw, explain=False):
        image = decode_image(raw)
        x = self.transform(image).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self.lock:
            with torch.inference_mode():
                probability = float(self.model(x).sigmoid().item())
            result = decision(probability, self.metadata["threshold"])
            if explain:
                result["gradcam"] = self._gradcam(image, x, result["class_name"] == "hot_dog")
        result["inference_ms"] = round((time.perf_counter() - started)*1000, 2)
        result["confidence_note"] = "Model score, not a calibrated guarantee. Unfamiliar images may still receive high confidence."
        return result

    def _gradcam(self, image, x, positive):
        activations = []
        hook = self.model.features[-1].register_forward_hook(lambda module, inputs, output: activations.append(output))
        try:
            with torch.enable_grad():
                logit = self.model(x.detach().requires_grad_(True)).flatten()[0]
                score = logit if positive else -logit
                feature = activations[0]
                gradients = torch.autograd.grad(score, feature)[0]
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                heatmap = torch.relu((weights * feature).sum(dim=1, keepdim=True))
                heatmap = F.interpolate(heatmap, (224, 224), mode="bilinear", align_corners=False)[0, 0]
                heatmap = heatmap.detach().cpu().numpy()
                heatmap = heatmap / max(float(heatmap.max()), 1e-8)
            # Overlay on the actual center crop seen by the model, not the uncropped original.
            crop = transforms.CenterCrop(224)(transforms.Resize(232)(image))
            from matplotlib import colormaps
            color = (colormaps["turbo"](heatmap)[:, :, :3] * 255).astype(np.uint8)
            overlay = Image.blend(crop, Image.fromarray(color), 0.42)
            stream = io.BytesIO()
            overlay.save(stream, format="PNG")
            return {"image": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
                    "target": "hot_dog" if positive else "not_hot_dog",
                    "note": "Highlights regions influencing the predicted class on the model's 224px center crop; not a segmentation or proof of reasoning."}
        finally:
            hook.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model", default="artifacts/model.pt")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    result = Predictor(args.model).predict_bytes(Path(args.image).read_bytes(), args.explain)
    if "gradcam" in result:
        image_data = result["gradcam"].pop("image").split(",", 1)[1]
        target = Path("gradcam.png")
        target.write_bytes(base64.b64decode(image_data))
        result["gradcam"]["saved_to"] = str(target)
    print(json.dumps(result, indent=2))
