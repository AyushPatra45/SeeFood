"use strict";
const $ = (id) => document.getElementById(id);
let selectedFile = null;
let previewURL = null;
let activeRequest = null;
let modelReady = false;
const percent = (value) => `${(value * 100).toFixed(1)}%`;

function errorMessage(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function resetResult() {
  $("result-content").hidden = true;
  $("result-empty").hidden = false;
  $("result-state").textContent = "AWAITING EVIDENCE";
  $("result-panel").classList.remove("negative");
}

function busy(isBusy) {
  document.body.classList.toggle("busy", isBusy);
  $("result-panel").setAttribute("aria-busy", String(isBusy));
  $("predict-button").disabled = isBusy || !selectedFile || !modelReady;
  $("predict-label").textContent = isBusy ? "Taking a closer look…" : "Is it a hotdog?";
  $("result-state").textContent = isBusy ? "LOOKING AT THE EVIDENCE" : $("result-state").textContent;
}

function selectFile(file) {
  if (!file) return;
  if (activeRequest) activeRequest.abort();
  activeRequest = null;
  busy(false);
  errorMessage("");
  resetResult();
  // Clear the previous selection so an invalid replacement cannot submit an old image.
  selectedFile = null;
  $("predict-button").disabled = true;
  $("upload-empty").hidden = false;
  $("upload-preview").hidden = true;
  $("file-row").hidden = true;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    errorMessage("Choose a JPG, PNG, or WebP image.");
    return;
  }
  if (!file.size || file.size > 8 * 1024 * 1024) {
    errorMessage("Choose a non-empty image smaller than 8 MiB.");
    return;
  }
  selectedFile = file;
  previewURL = URL.createObjectURL(file);
  $("preview-image").src = previewURL;
  $("preview-image").onerror = () => {
    selectedFile = null;
    $("predict-button").disabled = true;
    errorMessage("This image could not be read. Please choose another file.");
  };
  $("upload-empty").hidden = true;
  $("upload-preview").hidden = false;
  $("file-row").hidden = false;
  $("file-name").textContent = file.name;
  $("file-size").textContent = `${(file.size / 1024).toFixed(0)} KB`;
  $("predict-button").disabled = !modelReady;
}

for (const id of ["choose-button", "change-button"]) {
  $(id).addEventListener("click", () => { $("file-input").value = ""; $("file-input").click(); });
}
$("file-input").addEventListener("change", (event) => selectFile(event.target.files[0]));
for (const name of ["dragenter", "dragover"]) {
  $("dropzone").addEventListener(name, (event) => { event.preventDefault(); $("dropzone").classList.add("dragging"); });
}
for (const name of ["dragleave", "drop"]) {
  $("dropzone").addEventListener(name, (event) => { event.preventDefault(); $("dropzone").classList.remove("dragging"); });
}
$("dropzone").addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

$("predict-button").addEventListener("click", async () => {
  if (!selectedFile || !modelReady) return;
  errorMessage("");
  resetResult();
  const controller = new AbortController();
  activeRequest = controller;
  const form = new FormData();
  form.append("file", selectedFile);
  form.append("explain", $("explain-toggle").checked ? "true" : "false");
  busy(true);
  try {
    const response = await fetch("/predict", { method: "POST", body: form, signal: controller.signal });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "The image could not be processed.");
    if (activeRequest !== controller) return;
    $("result-empty").hidden = true;
    $("result-content").hidden = false;
    $("result-state").textContent = result.uncertain ? "A CLOSE CALL" : "PREDICTION COMPLETE";
    $("prediction").textContent = result.label;
    $("result-panel").classList.toggle("negative", result.class_name === "not_hot_dog");
    $("confidence").textContent = percent(result.confidence);
    $("confidence-fill").style.width = percent(result.confidence);
    $("hotdog-score").textContent = percent(result.probability_hot_dog);
    $("uncertainty-note").hidden = !result.uncertain;
    $("gradcam-section").hidden = !result.gradcam;
    if (result.gradcam) $("gradcam-image").src = result.gradcam.image;
    $("inference-time").textContent = `${result.inference_ms.toFixed(0)} ms${result.gradcam ? " incl. explanation" : " inference"}`;
  } catch (error) {
    if (activeRequest === controller && error.name !== "AbortError") {
      errorMessage(error.message || "Unable to connect. Check that the API is running.");
      $("result-state").textContent = "PLEASE TRY AGAIN";
    }
  } finally {
    if (activeRequest === controller) { activeRequest = null; busy(false); }
  }
});

async function loadStatus() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Health check failed");
    const health = await response.json();
    modelReady = health.model_loaded;
    $("model-status").classList.toggle("ready", modelReady);
    $("model-status").classList.toggle("error", !modelReady);
    $("status-text").textContent = modelReady ? "Model ready" : "Model unavailable";
    $("predict-button").disabled = !modelReady || !selectedFile;
    if (!modelReady) errorMessage("Model weights are missing. Follow the README to train or download them, then restart the server.");
  } catch {
    $("status-text").textContent = "API offline";
    $("model-status").classList.add("error");
    errorMessage("Cannot reach the API. Start the application using the README instructions.");
  }
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/metrics");
    if (!response.ok) throw new Error("No report");
    const report = await response.json();
    for (const key of ["accuracy", "precision", "recall", "f1"]) $("metric-" + key).textContent = percent(report[key]);
    $("test-note").textContent = `${report.test_size} held-out test images · Positive class: hotdog · Validation-selected decision threshold: ${report.threshold.toFixed(2)}`;
  } catch {
    $("test-note").textContent = "Evaluation report is unavailable. Run the evaluation command in the README to generate it.";
  }
}
loadStatus();
loadMetrics();
