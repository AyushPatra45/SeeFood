# Verification record

Verified locally on 21 October 2026 using Python 3.12.14, PyTorch 2.14.1, torchvision 0.29.1, macOS ARM64. Full installed versions are in `requirements-lock-macos.txt`.

## Completed checks

- Downloaded and extracted the actual public Kaggle ZIP; archive SHA256 recorded in `reports/data_audit.json`.
- Checked the archive's duplicate packaging, image decodability, class counts, exact/perceptual duplicate groups, and train/validation/test group isolation.
- Trained MobileNetV2 in both required stages on Apple MPS; saved the best validation-loss checkpoint, with explicit class mapping.
- Selected the classification threshold using validation F1, then evaluated the final checkpoint on all 500 official test images.
- Generated accuracy, precision, recall, F1, confusion matrix, learning curves, per-image predictions, and an error list. Reviewed six error images visually and documented apparent label noise and food-confusion examples.
- Verified the saved test metrics can be recomputed from the committed prediction report.
- Exported valid FP32 and static INT8 ONNX models. FP32 parity error is below `1e-4` on 20 validation images. INT8 calibration uses validation examples only.
- Benchmarked model-only CPU ONNX Runtime latency after warmup, and measured both exports on the fixed test set without retuning the threshold.
- Ran **23 automated tests successfully** in the local environment. One third-party Starlette warning recommends a future test-client transport migration; it does not affect the passing results.
- `python -m compileall -q seefood api scripts tests`, `node --check frontend/app.js`, `python -m pip check`, and `git diff --check` succeeded.
- Started the real FastAPI server and inspected the frontend in the in-app browser. Verified model-ready status, image selection/preview, visible loading state, actual POST prediction, returned score, Grad-CAM display, invalid-file rejection, recovery on a subsequent valid upload, and the uncertainty warning. `docs/prediction.png` is a real browser screenshot.
- Ran outside-dataset inference using public Wikimedia images. Their SHA256 values and nearest perceptual-hash distances are recorded in `reports/external_inference.json`; neither matches the dataset under the implemented duplicate checks. Raw external files are not redistributed.

## Results and material limits

The application uses the validation-selected threshold of 0.15: **87.4% accuracy, 81.06% precision, 97.6% recall, 88.57% F1**. The default-threshold baseline is also reported (91.4% accuracy); the operating threshold was not switched after viewing test results.

The external hotdog receives a 99.6% hotdog score. The external cat is misclassified under the recall-focused threshold at a 27.6% hotdog score and is flagged uncertain by the entropy heuristic. This is a known semantic failure, not a pipeline error. The cat image is [Cat03.jpg by Fir0002/Flagstaffotos](https://commons.wikimedia.org/wiki/File:Cat03.jpg), used locally for a smoke test; its image bytes are not included in this repository.

INT8 is 70.1% smaller, with negligible measured speed change and lower test accuracy (85.8%). The main application intentionally serves the original checkpoint. Confidence/entropy are not calibrated guarantees or robust OOD detection. Grad-CAM is not proof of reasoning. Hash heuristics do not establish absence of every duplicate or overlap with ImageNet pretraining.

Desktop browser behavior was inspected directly. Responsive CSS is included, but a separate physical mobile-device browser was not tested. Public hosting is not claimed; the saved-weight download option fulfills the submission requirement. GitHub Actions status is separate from these local checks.

## Docker

The final Docker image built successfully and passed a real HTTP smoke check on **Linux ARM64**, using Python 3.12.15, PyTorch 2.14.1+cpu, and torchvision 0.29.1+cpu. Verified model readiness, frontend, metrics, plot delivery, hotdog prediction plus Grad-CAM, the cat-image uncertainty warning, corrupt-file HTTP 422, and invalid-MIME HTTP 415. The test container was stopped afterward. See `reports/container_smoke.json` for the actual responses. Other container architectures and public hosting providers have not been tested.
