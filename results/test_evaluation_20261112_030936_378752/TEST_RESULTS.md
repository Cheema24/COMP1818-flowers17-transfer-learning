# Fixed-checkpoint test evaluation

Official split 1: 340 test images, 20 per class. No training or checkpoint changes occurred.

| Method | Test accuracy | Macro-F1 | Correct / 340 | Test loss |
|---|---:|---:|---:|---:|
| Frozen baseline | 92.35% | 0.9232 | 314 | 0.3156 |
| Full LP-FT | 92.65% | 0.9265 | 315 | 0.2953 |
| Final-block extension | 92.94% | 0.9290 | 316 | 0.3113 |

Rows in confusion matrices are true classes; columns are predictions. Labels are 0-16.
Learning curves use validation data. Checkpoints were selected by validation loss.
One split and one seed cannot establish superiority. Do not tune against test scores.
Training times include stage 1. All deployed models have the same architecture.
