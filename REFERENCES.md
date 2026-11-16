# Source references

Read these original sources and format citations according to the Moodle template; this list is not a finished report bibliography.

- Kumar, A., Raghunathan, A., Jones, R., Ma, T. and Liang, P. (2022). *Fine-Tuning Can Distort Pretrained Features and Underperform Out-of-Distribution*. ICLR. https://arxiv.org/abs/2202.10054
- Oxford Visual Geometry Group. *17 Category Flower Dataset*. Dataset, original paper links and official split files: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/ . Original source README and downloaded-file hashes are preserved in data/.
- Torchvision. *mobilenet_v3_small*. Architecture, pretrained weights, preprocessing and link to the original MobileNetV3 paper: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html . Model weight URL and checksum are in models/source.json.

The MobileNet implementation is provided by torchvision. The project scripts load its architecture and weights rather than implementing MobileNet from scratch. The LP-FT strategy is from Kumar et al.; final-block-only adaptation is the investigated modification.
