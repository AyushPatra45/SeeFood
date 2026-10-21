"""Serve a real trained model and static interface from the same origin."""
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from seefood.predict import MAX_BYTES, Predictor

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path = Path(model_path or os.getenv("MODEL_PATH", ROOT / "artifacts/model.pt"))
        app.state.predictor = None
        if path.exists():
            # Fail visibly for a corrupt/incompatible model rather than return fake predictions.
            app.state.predictor = await run_in_threadpool(Predictor, path)
        else:
            logger.warning("Model missing: %s. /predict will return 503.", path)
        yield
        app.state.predictor = None

    app = FastAPI(title="SeeFood API", version="1.0.0", lifespan=lifespan,
                  description="MobileNetV2 hotdog classifier. Model scores are not calibrated guarantees.")
    app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ROOT / "frontend/index.html")

    @app.get("/health")
    def health():
        return {"status": "ready" if app.state.predictor else "model_missing", "model_loaded": app.state.predictor is not None}

    @app.get("/api/metrics")
    def metrics():
        path = ROOT / "reports/metrics.json"
        if not path.exists():
            raise HTTPException(404, "Evaluation report is not available yet.")
        return json.loads(path.read_text())

    @app.get("/api/model")
    def model_info():
        if app.state.predictor is None:
            raise HTTPException(503, "Model is missing. Follow the README to train or download weights.")
        metadata = app.state.predictor.metadata
        return {"architecture": "MobileNetV2", "class_to_idx": metadata["class_to_idx"],
                "threshold": metadata["threshold"], "image_size": 224,
                "best_stage": metadata["best_stage"], "best_epoch": metadata["best_epoch"]}

    @app.get("/api/plots/{name}")
    def plot(name: str):
        if name not in {"confusion_matrix", "training_curves"}:
            raise HTTPException(404, "Unknown plot")
        path = ROOT / "reports" / f"{name}.png"
        if not path.exists():
            raise HTTPException(404, "Plot is not available yet.")
        return FileResponse(path)

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), explain: bool = Form(False)):
        if app.state.predictor is None:
            raise HTTPException(503, "Model is missing. Follow the README to train or download weights.")
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            await file.close()
            raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")
        chunks, size = [], 0
        try:
            while chunk := await file.read(65536):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Image must be smaller than 8 MiB.")
                chunks.append(chunk)
            try:
                return await run_in_threadpool(app.state.predictor.predict_bytes, b"".join(chunks), explain)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        finally:
            await file.close()

    return app


app = create_app()
