# -*- coding: utf-8 -*- 
import io
from PIL import Image
from app import model, imagenet_class_index, my_preprocess
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy
import threading
from datetime import datetime
import time

def log_print(message):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    thread_id = threading.get_ident()
    prefijo = f"[{time}] [{thread_id}]"
    print(prefijo, message)

def transform_image(image_bytes):
    image_bytes2 = io.BytesIO(image_bytes)
    image = Image.open(image_bytes2)
    if image.mode == "RGBA":
        log_print("imagen tiene transparencia, usar fondo blanco")
        background = Image.new(image.mode, image.size, (255,255,255,0))
        alpha_composite = Image.alpha_composite(background, image)
        image = alpha_composite.convert('RGB')
    # aplicar las funciones
    image_tensor = my_preprocess(image).unsqueeze(0)
    return image_tensor


def get_prediction(image_bytes):
    t0 = time.time()
    log_print("leyendo imagen")
    image_tensor = transform_image(image_bytes)
    log_print("forward de tensor {} en la red".format(image_tensor.size()))
    outputs = model.forward(image_tensor)
    probabilities = F.softmax(outputs, dim=1)
    top_scores, top_positions = torch.topk(probabilities, 3)
    tiempo = round(time.time() - t0, 3)
    clases = []
    for i in range(3):
        score = round(top_scores[0][i].item(), 3)
        position = str(top_positions[0][i].item())
        clase_id, clase_nombre = imagenet_class_index[position]
        log_print("clase: {} {} score: {:.3f}".format(clase_id, clase_nombre, score))
        clases.append({'clase_id': clase_id, 'clase_nombre': clase_nombre, 'score': score})
    return clases, tiempo
