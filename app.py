# -*- coding: utf-8 -*- 
import json
from torchvision import models
from flask import Flask
import torchvision.transforms as transforms

app = Flask(__name__)

# Ver otros modelos en https://docs.pytorch.org/vision/main/models.html
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
model.eval()

# se cada modelo tiene una función con su pre-procesamiento, pero no funciona bien
# my_preprocess = weights.transforms()
my_preprocess = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])

# leer los nombres de las clases
FILENAME_IMAGENET_CLASSES = 'imagenet_class_index.json'
imagenet_class_index = json.load(open(FILENAME_IMAGENET_CLASSES))
