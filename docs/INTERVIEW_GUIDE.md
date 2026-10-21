# Explain SeeFood from bytes to prediction

Use this as a study guide, not a memorized script. Open the source alongside each explanation and reproduce the behavior yourself.

## A 60-second project explanation

“SeeFood classifies an uploaded image as hotdog or not hotdog. I use an ImageNet-pretrained MobileNetV2 and replace its classifier with one output logit. The training pipeline freezes the backbone first, then fine-tunes its final blocks at a smaller learning rate. Data preparation preserves the official test split and keeps detected duplicate groups out of separate splits. FastAPI validates uploads, applies the same preprocessing as evaluation, and returns a label and score. The interface also shows Grad-CAM. I report held-out precision, recall, F1, accuracy, and a confusion matrix, including the threshold tradeoff and quantization results.”

Adapt this explanation after tracing the relevant source files yourself.

## Trace one image

1. The browser creates a multipart request containing the file and `explain` flag.
2. FastAPI checks MIME type, reads up to the byte limit, and closes the upload afterward.
3. Pillow checks the actual image format, dimensions, and decodability. EXIF orientation is applied; grayscale/alpha inputs become RGB.
4. Resize the shorter edge to 232 and take a center crop of 224×224. This preserves aspect ratio but can crop objects near the edge.
5. `ToTensor` changes H×W×C integer pixels into a C×H×W float tensor in [0,1]. Normalize each channel: `(pixel - mean) / std`.
6. Add the batch dimension: `[1,3,224,224]`.
7. The network returns one logit `z`; sigmoid produces `p = 1/(1+exp(-z))`.
8. If `p >= 0.15`, return hotdog. Otherwise return not hotdog. The displayed class score is `p` or `1-p`, respectively.

Find the implementation in `seefood/predict.py` and `seefood/data.py`.

## What does a convolution do?

A small learned filter slides over the image or feature map. At each location it multiplies corresponding values and sums them to produce a feature response. Sharing the same filter across locations reduces parameters and lets the same visual pattern be detected in different positions. Early features often respond to edges and textures; later combinations become more task-specific.

MobileNetV2 uses efficient depthwise convolutions followed by pointwise 1×1 convolutions, inverted residual blocks, and linear bottlenecks. Depthwise convolution processes each channel separately; pointwise convolution mixes channels. Explain ordinary convolution first before discussing these optimizations.

## Why transfer learning?

We have only 397 training examples after cleaning/splitting. Training a large CNN from scratch would require learning all visual features from this small sample. ImageNet weights offer a useful initialization. The 1,281-parameter head is small enough to train first; then the later backbone blocks adapt to the new task.

Freezing means `requires_grad=False`: those weights do not receive parameter gradients or optimizer updates. Fine-tuning unfreezes selected parameters and uses a smaller learning rate to avoid drastic changes to useful pretrained features. BatchNorm remains fixed because small datasets can produce noisy running-statistic updates.

## Why this loss and optimizer?

Binary cross-entropy for label `y` and score `p` is:

`L = -[y log(p) + (1-y) log(1-p)]`

A confident wrong prediction has a high loss. The model emits a raw logit and PyTorch's `BCEWithLogitsLoss` handles the sigmoid/log terms with a numerically stable formulation. Applying sigmoid before this loss would be incorrect.

Backpropagation applies the chain rule through the network to compute derivatives of loss with respect to trainable weights. `loss.backward()` calculates gradients; `optimizer.step()` updates weights; `zero_grad()` clears the previous step's accumulated gradients. AdamW uses adaptive gradient statistics and decoupled weight decay. A learning rate controls update size.

## Training versus evaluation mode

`model.train()` enables dropout; `model.eval()` disables it and uses BatchNorm running statistics. In this project BatchNorm is explicitly held in evaluation mode during training too. `torch.inference_mode()` turns off autograd bookkeeping for normal inference; evaluation mode alone does not disable gradients. Grad-CAM deliberately enables gradients while keeping dropout disabled.

## What is overfitting?

A model can learn patterns specific to the training examples rather than generalizable visual features. A widening gap between training improvement and validation performance is one sign. Here augmentation, dropout, weight decay, early stopping, and restricted fine-tuning help. Training accuracy uses augmented images and dropout, while validation uses deterministic images, so their curves are not directly identical conditions.

## Why three splits?

Training updates weights. Validation selects checkpoint, hyperparameters, and threshold. Test measures final performance after those decisions. Tuning based on test mistakes would make the test less independent. Duplicates across splits can create the appearance of generalization without testing new examples.

The Kaggle ZIP has the same tree twice. The loader selects one tree. It also excludes two training images connected to official-test duplicate groups and uses group-aware validation splitting. Hashing cannot prove all semantic overlaps are absent.

## Read the confusion matrix

Rows are actual classes; columns are predictions. Order: not hotdog, hotdog.

```text
                     Predicted not   Predicted hotdog
Actual not hotdog        193                57
Actual hotdog              6               244
```

- Accuracy: `(193+244)/500 = 87.4%`.
- Precision: `244/(244+57) = 81.06%`.
- Recall: `244/(244+6) = 97.6%`.
- F1: `2*precision*recall/(precision+recall) = 88.57%`.

A false positive calls non-hotdog food a hotdog. A false negative misses a real hotdog according to the dataset labels.

## Why is the threshold 0.15 rather than 0.5?

It maximized F1 on the 99-image validation set. Lowering the threshold tends to increase recall and false positives. On the test set the 0.5 baseline actually had higher accuracy and F1. This illustrates validation threshold overfitting and sampling differences. We report both and keep the preselected threshold; choosing the best test threshold afterward would be leakage. A stronger follow-up would use a larger representative validation set or cross-validation, then assess once on a new untouched test set.

A sigmoid score is not automatically a calibrated probability. Threshold selection changes the decision rule; it does not calibrate probabilities. The close-call band and binary-entropy cutoff are heuristics. Entropy is `-[p log(p)+(1-p) log(1-p)]`; a cutoff of 0.50 nats flags ambiguous scores. They cannot reliably detect a dog in a hotdog costume or arbitrary unfamiliar images. A recorded external cat photo is misclassified at the recall-focused threshold and flagged uncertain; describe this limitation directly.

## How does Grad-CAM work?

Capture the final convolutional feature maps. Differentiate the selected class logit with respect to those maps. Average the gradients spatially to weight each feature map, take their weighted sum, and apply ReLU. Resize and normalize the result for visualization. For a not-hotdog explanation, use the negative of the hotdog logit.

The overlay uses the actual center crop so coordinates align. Grad-CAM shows influential regions at a coarse resolution; it is not an object detector, segmentation mask, or proof of causal reasoning. Hooks are removed in `finally`, and inference is locked to prevent concurrent explanations from mixing activations.

## What did quantization achieve?

Static INT8 QDQ uses representative validation images to estimate activation ranges. It reduces model storage from 8.87 MB to 2.65 MB, approximately 70%. In this benchmark the speed change was negligible and test accuracy fell from 87.4% to 85.8%. The application therefore uses the original model. Hardware, runtime kernels, and conversion overhead determine whether quantization actually speeds inference up.

## Be ready to demonstrate

Start the app; upload a new hotdog and a non-hotdog; disable and enable Grad-CAM; try a corrupt file; show the test confusion matrix. Open a recorded false positive and false negative from the downloaded dataset. Describe what you observe, and distinguish observation from guesses about the cause. Some labels appear noisy; do not silently repair test labels to improve a score.

Run `python -m pytest -q`, explain one test, then point to the exact code for normalization, freezing, loss, threshold selection, and upload decoding. That is stronger evidence of understanding than memorizing the accuracy.
