"""Check real checkpoint contracts, leakage boundaries, upload failures, and explanations."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

from api.main import ROOT, create_app
from seefood.data import image_transform
from seefood.evaluate import metric_summary
from seefood.model import CLASS_TO_IDX, build_model, configure_stage
from seefood.predict import MAX_BYTES, Predictor, decode_image, decision


def png_bytes(mode="RGB", size=(300, 250)):
    image = Image.new(mode, size, "orange" if mode == "RGB" else 120)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture(scope="module")
def predictor():
    return Predictor(ROOT / "artifacts/model.pt")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ROOT / "artifacts/model.pt")) as test_client:
        yield test_client


def test_manifest_does_not_count_packaged_copy_or_leak_groups():
    rows = json.loads((ROOT / "reports/manifest.json").read_text())
    groups = {s: {r["group"] for r in rows if r["split"] == s} for s in ["train", "val", "test"]}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert sum(r["split"] == "test" for r in rows) == 500
    assert len({r["path"] for r in rows}) == len(rows)
    assert all(not r["path"].startswith("seefood/") for r in rows)
    for split in groups:
        assert {r["label"] for r in rows if r["split"] == split} == {0, 1}


def test_freeze_then_finetune_only_intended_parameters():
    model = build_model()
    configure_stage(model, "head")
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    configure_stage(model, "finetune")
    assert all(not p.requires_grad for p in model.features[:15].parameters())
    assert any(p.requires_grad for p in model.features[15:].parameters())
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_decoding_accepts_color_modes_and_preprocessing_is_deterministic(mode):
    image = decode_image(png_bytes(mode))
    assert image.mode == "RGB"
    transform = image_transform()
    assert torch.equal(transform(image), transform(image))
    assert transform(image).shape == (3, 224, 224)
    assert torch.isfinite(transform(image)).all()


@pytest.mark.parametrize("raw", [b"", b"not an image", b"<script>alert(1)</script>", b"x" * (MAX_BYTES+1)])
def test_invalid_image_bytes_are_rejected(raw):
    with pytest.raises(ValueError):
        decode_image(raw)


def test_pixel_limit():
    with pytest.raises(ValueError, match="megapixel"):
        decode_image(png_bytes("L", (4001, 4000)))


def test_entropy_warning_does_not_masquerade_as_an_ood_detector():
    ambiguous = decision(0.276, 0.15)
    assert ambiguous["class_name"] == "hot_dog"
    assert ambiguous["uncertain"]
    assert "high_binary_entropy" in ambiguous["uncertainty_reasons"]
    assert decision(0.5, 0.15)["binary_entropy_nats"] == pytest.approx(np.log(2))
    assert not decision(0.99, 0.15)["uncertain"]


def test_prediction_matches_direct_model_and_saved_threshold(predictor):
    raw = png_bytes()
    result = predictor.predict_bytes(raw)
    tensor = image_transform()(decode_image(raw)).unsqueeze(0)
    with torch.inference_mode():
        direct = predictor.model(tensor).sigmoid().item()
    assert result["probability_hot_dog"] == pytest.approx(direct, abs=1e-6)
    assert result["class_name"] == ("hot_dog" if direct >= result["threshold"] else "not_hot_dog")
    assert result["confidence"] == pytest.approx(direct if result["class_name"] == "hot_dog" else 1-direct)
    assert predictor.metadata["class_to_idx"] == CLASS_TO_IDX


def test_gradcam_and_concurrent_inference_leave_no_hooks(predictor):
    raw = png_bytes()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: predictor.predict_bytes(raw, explain=True), range(2)))
    import base64
    for result in results:
        encoded = result["gradcam"]["image"].split(",")[1]
        heatmap = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert heatmap.size == (224, 224)
    assert results[0]["probability_hot_dog"] == results[1]["probability_hot_dog"]
    assert len(predictor.model.features[-1]._forward_hooks) == 0


def test_health_ui_and_reports(client):
    assert client.get("/health").json()["model_loaded"] is True
    assert "seefood" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/model").json()["class_to_idx"] == CLASS_TO_IDX
    assert client.get("/api/metrics").json()["test_size"] == 500
    assert client.get("/api/plots/confusion_matrix").status_code == 200
    assert client.get("/api/plots/unknown").status_code == 404


def test_api_valid_image_and_explanation(client):
    response = client.post("/predict", files={"file": ("sample.png", png_bytes(), "image/png")}, data={"explain": "true"})
    assert response.status_code == 200
    assert response.json()["gradcam"]["image"].startswith("data:image/png;base64,")
    assert 0 <= response.json()["confidence"] <= 1


@pytest.mark.parametrize("mime,raw,status", [("text/plain", b"hello", 415), ("image/png", b"fake png", 422),
                                              ("image/png", b"", 422), ("image/png", b"0"*(MAX_BYTES+1), 413)])
def test_api_rejects_bad_uploads(client, mime, raw, status):
    response = client.post("/predict", files={"file": ("image.png", raw, mime)})
    assert response.status_code == status
    assert isinstance(response.json()["detail"], str)


def test_api_missing_file(client):
    assert client.post("/predict").status_code == 422


def test_missing_model_is_explicit(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt")) as client:
        assert client.get("/health").json()["status"] == "model_missing"
        assert client.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")}).status_code == 503


def test_saved_test_report_can_be_recomputed():
    rows = json.loads((ROOT / "reports/test_predictions.json").read_text())
    report = json.loads((ROOT / "reports/metrics.json").read_text())
    expected = metric_summary([r["true_label"] for r in rows], [r["probability_hot_dog"] for r in rows], report["threshold"])
    for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
        assert expected[key] == report[key]


def test_actual_training_ran_both_stages_and_selected_threshold_on_validation(predictor):
    history = json.loads((ROOT / "reports/history.json").read_text())
    assert {r["stage"] for r in history} == {"head", "finetune"}
    assert history[-1]["trainable_parameters"] > history[0]["trainable_parameters"]
    val = json.loads((ROOT / "reports/validation_predictions.json").read_text())
    from seefood.train import select_threshold
    assert select_threshold(np.array([r["label"] for r in val]), np.array([r["probability_hot_dog"] for r in val])) == predictor.metadata["threshold"]
