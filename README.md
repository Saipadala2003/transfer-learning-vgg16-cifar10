# Transfer Learning using VGG16 on CIFAR-10

## Overview
This project implements transfer learning using the pretrained VGG16 convolutional neural network on the CIFAR-10 image classification dataset. A custom CNN baseline is trained first, followed by VGG16 transfer learning and fine-tuning. The experiments compare classification accuracy, loss, and training time.

## Objective
- Implement transfer learning using a pretrained VGG16 model.
- Freeze pretrained layers and train a new classification head.
- Fine-tune selected VGG16 layers with a lower learning rate.
- Compare the transfer-learning model with a custom CNN baseline.
- Analyze accuracy, loss, training time, and classification performance.

## Dataset
**CIFAR-10** contains 60,000 colour images of size 32×32 across 10 classes. The standard split used in this experiment contains 50,000 training images and 10,000 test images.

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

For VGG16, the CIFAR-10 images are resized to 96×96×3 to match the transfer-learning input pipeline used in the notebook. VGG16 uses ImageNet pretrained weights with the original classification head removed.

## Methodology
1. Load CIFAR-10 and preprocess the images.
2. Train a custom CNN baseline.
3. Load VGG16 with ImageNet pretrained weights and `include_top=False`.
4. Freeze the VGG16 base and train a new classification head.
5. Unfreeze the last four VGG16 layers and fine-tune using a learning rate of `1e-5`.
6. Evaluate the final model on the CIFAR-10 test set.
7. Compare the baseline and transfer-learning results using plots and a confusion matrix.

## Results
| Model | Test Accuracy | Test Loss | Training Time |
|---|---:|---:|---:|
| Custom CNN baseline | 67.41% | 0.9160 | 25.55 s |
| VGG16 transfer learning | 89.88% | 0.3471 | 985.83 s |
| VGG16 + fine-tuning | **93.33%** | **0.2998** | 283.71 s fine-tuning |

The final VGG16 model improves test accuracy by **25.92 percentage points** over the custom CNN baseline.

The total VGG16 pipeline time, including initial transfer learning and fine-tuning, was approximately **1269.54 seconds** in the recorded experiment.

## Experimental Environment
- TensorFlow 2.20.0
- Google Colab
- NVIDIA T4 GPU
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Repository Contents
- `Task_2_Transfer_Learning_VGG16_CIFAR10.ipynb` — complete Jupyter/Colab notebook with code, training, evaluation, and outputs.
- `Task_2_Transfer_Learning_VGG16_CIFAR10_Report.pdf` — academic report containing methodology, source code, screenshots, results, observations, and conclusion.
- `results/` — accuracy, loss, training-time, sample-data, and confusion-matrix figures.

## Key Observation
Transfer learning substantially improved classification accuracy compared with the custom CNN baseline. Fine-tuning the final four VGG16 layers further improved the recorded accuracy from 89.88% to 93.33%, demonstrating the benefit of adapting pretrained visual features to the CIFAR-10 task.

## Conclusion
The experiment demonstrates that pretrained deep convolutional features can provide a strong starting point for image classification. VGG16 achieved substantially better test accuracy than the custom CNN baseline, and selective fine-tuning produced the best recorded result of 93.33%.

## Author
**Saikumar Ganesh Padala**

MSc Artificial Intelligence

## License
This repository is intended for academic and educational use.
