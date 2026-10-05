# -*- coding: utf-8 -*- 
import io
from PIL import Image
from app import model, imagenet_class_index, my_preprocess
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy
import cv2

def transform_image(image_bytes):
    image_bytes2 = io.BytesIO(image_bytes)
    image = Image.open(image_bytes2)
    if image.mode == "RGBA":
        print("imagen tiene transparencia, usar fondo blanco")
        background = Image.new(image.mode, image.size, (255,255,255,0))
        alpha_composite = Image.alpha_composite(background, image)
        image = alpha_composite.convert('RGB')
    # aplicar las funciones
    image_tensor = my_preprocess(image).unsqueeze(0)
    return image_tensor


def get_prediction(image_bytes):
    print("leyendo imagen")
    image_tensor = transform_image(image_bytes)
    print("forward de tensor {} en la red".format(image_tensor.size()))
    outputs = model.forward(image_tensor)
    probabilities = F.softmax(outputs, dim=1)
    score_tensor, position_tensor = probabilities.max(1)
    score = score_tensor.item()
    position = position_tensor.item()
    print("max-position: {} max-score: {:.3f}".format(position, score))
    array = imagenet_class_index[str(position)]
    clase_id = array[0]
    clase_nombre = array[1]
    print("clase: {} {}".format(clase_id, clase_nombre))
    # ¿cuanto tiempo demoró en predecir? devolverlo
    return clase_id, clase_nombre
