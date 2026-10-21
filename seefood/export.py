"""Export FP32 and static INT8 ONNX; calibrate on validation images and benchmark honestly."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import torch

from seefood.data import ImageManifest
from seefood.evaluate import metric_summary
from seefood.model import load_checkpoint


class ImageCalibration(CalibrationDataReader):
    def __init__(self, dataset, limit=64):
        indices = np.random.default_rng(42).permutation(len(dataset))[:limit]
        self.items = iter({"image": dataset[int(i)][0].unsqueeze(0).numpy()} for i in indices)

    def get_next(self):
        return next(self.items, None)


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def benchmark(runtime, example, repeats=50):
    for _ in range(10):
        runtime.run(None, {"image": example})
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        runtime.run(None, {"image": example})
        times.append((time.perf_counter()-start)*1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "repeats": repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--model", default="artifacts/model.pt")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.model)
    output, reports = Path(args.model).parent, Path(args.reports)
    fp32, int8 = output / "model.onnx", output / "model.int8.onnx"
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, dummy, str(fp32), input_names=["image"], output_names=["logit"],
                      opset_version=18, dynamo=False, dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}})
    onnx.checker.check_model(str(fp32))
    val = ImageManifest(args.data, reports / "manifest.json", "val")
    quantize_static(str(fp32), str(int8), ImageCalibration(val), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv", "Gemm", "MatMul"])
    onnx.checker.check_model(str(int8))
    runtimes = {"fp32": session(fp32), "int8": session(int8)}
    example = val[0][0].unsqueeze(0).numpy()
    equivalence = []
    with torch.inference_mode():
        for i in range(min(20, len(val))):
            tensor = val[i][0].unsqueeze(0)
            expected = float(model(tensor).sigmoid().item())
            logit = float(runtimes["fp32"].run(None, {"image": tensor.numpy()})[0].flatten()[0])
            equivalence.append(abs(expected - 1/(1+np.exp(-logit))))
    if max(equivalence) > 1e-4:
        raise RuntimeError("FP32 ONNX export does not match PyTorch within tolerance")
    report = {"platform": platform.platform(), "processor": platform.machine(), "provider": "CPUExecutionProvider",
              "threads": 4, "batch_size": 1, "input_shape": [1, 3, 224, 224],
              "timing_scope": "Model only; excludes decoding, preprocessing, network, and Grad-CAM.",
              "quantization": "Static QDQ INT8 weights / UINT8 activations; up to 64 validation calibration images; per-channel weights",
              "fp32_max_probability_error_vs_pytorch": max(equivalence),
              "fp32": {"bytes": fp32.stat().st_size, **benchmark(runtimes["fp32"], example)},
              "int8": {"bytes": int8.stat().st_size, **benchmark(runtimes["int8"], example)}}
    report["size_reduction_percent"] = 100*(1-report["int8"]["bytes"]/report["fp32"]["bytes"])
    report["speedup_ratio"] = report["fp32"]["median_ms"]/report["int8"]["median_ms"]
    report["speed_note"] = "A ratio above 1 means faster INT8; below 1 means INT8 was slower on this machine."
    # Report the quantization tradeoff without selecting a new threshold on test data.
    test = ImageManifest(args.data, reports / "manifest.json", "test")
    targets, scores = [], {"fp32": [], "int8": []}
    for i in range(len(test)):
        tensor, label = test[i]
        targets.append(label)
        for name, runtime in runtimes.items():
            logit = float(runtime.run(None, {"image": tensor.unsqueeze(0).numpy()})[0].flatten()[0])
            scores[name].append(float(1/(1+np.exp(-logit))))
    for name in scores:
        report[name]["test_metrics"] = metric_summary(targets, scores[name], checkpoint["threshold"])
    (reports / "optimization.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
