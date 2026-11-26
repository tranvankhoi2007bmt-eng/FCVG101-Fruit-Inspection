"""Small CNN shared by training and deployment."""
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps

LABELS = ['fresh', 'rotten']
SIZE = 128

class FruitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((2,2)))
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(512,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,2))
    def forward(self,x): return self.classifier(self.features(x))

def transform(image, augment=False):
    import random
    image = ImageOps.exif_transpose(image).convert('RGB').resize((SIZE,SIZE))
    if augment:
        if random.random() < .5: image=ImageOps.mirror(image)
        image=image.rotate(random.uniform(-15,15), resample=Image.Resampling.BILINEAR,fillcolor=(127,127,127))
        from PIL import ImageEnhance
        image=ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
    a=np.asarray(image,dtype=np.float32)/255.0
    return torch.from_numpy(((a-.5)/.5).transpose(2,0,1).copy())

def predict(model, image):
    model.eval()
    with torch.inference_mode():
        p=model(transform(image).unsqueeze(0).to(next(model.parameters()).device)).softmax(dim=1)[0].cpu()
    return {label:float(p[i]) for i,label in enumerate(LABELS)}
