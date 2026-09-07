# Experimental Results

## Recorded Performance

| Model | Test Accuracy | Test Loss | Time |
|---|---:|---:|---:|
| Custom CNN baseline | 67.41% | 0.9160 | 25.55 s |
| VGG16 transfer learning | 89.88% | 0.3471 | 985.83 s |
| VGG16 + fine-tuning | **93.33%** | **0.2998** | 283.71 s fine-tuning |

**Accuracy improvement over baseline:** 25.92 percentage points.

**Total VGG16 pipeline time:** approximately 1269.54 seconds.

## Fine-Tuning Configuration

- Pretrained model: VGG16
- Pretrained weights: ImageNet
- Input size: 96 × 96 × 3
- Base model: `include_top=False`
- Fine-tuned layers: last 4 VGG16 layers
- Fine-tuning learning rate: `0.00001`
- Fine-tuning epochs: 3

## Evaluation

The final model achieved approximately 93% overall accuracy on the 10,000-image CIFAR-10 test set. A confusion matrix and classification report were generated in the notebook.
