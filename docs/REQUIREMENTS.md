# Assignment coverage

The source PDF is the SeeFood Technical Department Recruitment assignment. Its second-year criteria apply. This checklist maps requirements to concrete implementation; bonus items are distinguished explicitly.

| Requirement | Implementation / evidence |
|---|---|
| Public Kaggle binary-image dataset | `scripts/download_data.py`; source and archive hash in `reports/data_audit.json` |
| Load, resize, normalize | `seefood/data.py`; shared PyTorch ImageNet transforms |
| Proper train/validation/test split | `seefood/prepare.py`; 397/99/500 images; group isolation tested |
| Train a binary classifier | `seefood/train.py`; committed `artifacts/model.pt` |
| Accuracy, precision, recall, F1 | `reports/metrics.json`; displayed by frontend |
| Confusion matrix | `reports/confusion_matrix.png` |
| New image prediction and confidence | `seefood/predict.py`; `POST /predict`; image-upload UI |
| PyTorch or TensorFlow | PyTorch + torchvision |
| Modular source structure | Separate data, model, training, evaluation, inference, export, API, frontend |
| Pretrained backbone and custom head | MobileNetV2 ImageNet V2 + dropout + one-logit linear head |
| Demonstrate freezing and fine-tuning | `configure_stage`; both stages and trainable parameter counts in `reports/history.json` |
| REST API and clean frontend | FastAPI + responsive HTML/CSS/JavaScript |
| Invalid files and loading states | HTTP 415/422/413/503, frontend feedback, busy state; tested |
| Public GitHub source | This repository |
| `requirements.txt` | Root requirements plus optional dev/export requirements |
| Do not push raw dataset | `.gitignore`, `.dockerignore`; download instructions provided |
| Live app OR downloadable weights | Three committed downloadable model files linked from README; public hosting is not required for this submission option |
| README setup, download, inference | README quick start and reproduction commands |
| Architecture and hyperparameter rationale | README and `docs/INTERVIEW_GUIDE.md` |
| Real challenge and solution | Archive duplication / split contamination, label noise, validation-threshold tradeoff |
| Bonus: augmentation | Training-only crop, flip, rotation, and color jitter |
| Bonus: learning curves | `reports/training_curves.png` |
| Bonus: interactive UI | Full upload and result interface |
| Bonus: Grad-CAM | Predicted-class heatmap aligned to the model crop |
| Bonus: optimization | FP32/INT8 ONNX plus size, timing, parity, and accuracy checks |
| Bonus: threshold tuning / entropy | Validation F1 selection and boundary/entropy warning; no claim of robust OOD detection or calibrated probabilities |

See `docs/VERIFICATION.md` for actual verification results and limits.
