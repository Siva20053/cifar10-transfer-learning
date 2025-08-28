# Transfer Learning on CIFAR-10 (ResNet-18, PyTorch)

This repository contains an experiment with **transfer learning** on the CIFAR‑10 dataset using a pretrained **ResNet‑18** model in PyTorch. The project demonstrates how to fine‑tune a pretrained backbone, evaluate on CIFAR‑10, and save the trained model.

---

## Overview

* **Framework:** PyTorch (torch, torchvision)
* **Dataset:** CIFAR‑10 (60,000 32×32 images, 10 classes)
* **Backbone:** ResNet‑18 pretrained on ImageNet
* **Modification:** Final fully‑connected layer replaced to predict 10 classes
* **Training setup:**

  * Epochs: 5
  * Batch size: 64
  * Optimizer: Adam (`lr=0.001`)
  * Loss: CrossEntropyLoss
* **Result:** \~**80.38%** test accuracy in 5 epochs
* **Model file:** `cifar_resnet18.pth`

---

## Results

**Training log:**

```
Epoch :1 ,Loss :0.5738 ,Accuracy :79.98%
Epoch :2 ,Loss :0.5626 ,Accuracy :80.49%
Epoch :3 ,Loss :0.5606 ,Accuracy :80.59%
Epoch :4 ,Loss :0.5516 ,Accuracy :80.86%
Epoch :5 ,Loss :0.5491 ,Accuracy :80.68%

Test Accuracy : 80.38%
```

Example inference (notebook output): Predicted **ship**, Ground truth **ship** ✅

---

## Installation & Usage

1. Clone this repository:

```bash
git clone https://github.com/Siva20053/cifar10-transfer-learning.git
cd cifar10-transfer-learning
```

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate    # Windows
```

3. Install dependencies:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

4. Run the notebook:

```bash
jupyter notebook CIFAR10tf.ipynb
```

---

## Inference Example

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('cifar_resnet18.pth', map_location='cpu'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

img = Image.open('your_image.png').convert('RGB')
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    pred = outputs.argmax(dim=1).item()

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print('Predicted:', classes[pred])
```

---

## Possible Improvements

* Add **data augmentation** (RandomCrop, RandomHorizontalFlip, ColorJitter)
* Train for longer with **LR scheduling** (CosineAnnealing, ReduceLROnPlateau)
* Fine‑tune more layers at lower learning rates
* Try **stronger backbones** (ResNet‑34, ResNet‑50, EfficientNet, MobileNet)
* Apply regularization techniques (Mixup, CutMix)

---

## Files

* `CIFAR10tf.ipynb` — notebook with full workflow
* `cifar_resnet18.pth` — saved trained model weights

---

## License

MIT © Sivaramakrishna Reddy
