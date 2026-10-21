# Medical CNN Architecture

## Pipeline
1. Upload a local image.
2. Convert to RGB, resize to 224×224, tensorize, and normalize.
3. Three convolutional blocks extract features using convolution, batch normalization, ReLU, max pooling, and spatial dropout.
4. Global average pooling reduces the feature map.
5. A dropout-regularized fully connected classifier returns one logit per class.
6. Softmax scores are shown as model outputs, never as a clinical diagnosis.

## Regularization
- Dropout and Dropout2d reduce co-adaptation.
- AdamW weight decay penalizes large parameter values.
- Random flips and small rotations augment training images.
- Batch normalization stabilizes activations.
- Early stopping keeps the checkpoint with the best validation loss.

## Dataset contract
Use a properly licensed, de-identified dataset with documented provenance and labels. Expected layout: data/medical_images/train/<class_name>/ and data/medical_images/val/<class_name>/. Keep patient-level splits separate to reduce data leakage.

## Evaluation before research claims
Report held-out test results, class-wise precision/recall/F1, confusion matrix, sensitivity, specificity, calibration, confidence intervals, and subgroup performance. Never present demo or training metrics as clinical evidence. Follow applicable ethics, privacy, and clinical review requirements.
