# -*- coding: utf-8 -*- 
from app import app
from utils import get_prediction, log_print
from flask import Flask, jsonify, request


@app.route('/ejercicio2/app1-ia/predict', methods=['POST'])
def predict():
    api_key = request.headers.get('X-API-KEY')
    if api_key != 'Secreto_ejercicio2_202608101300':
        return jsonify({'error': 'Acceso no autorizado'}), 403
    file = request.files['file']
    file_bytes = file.read()
    log_print("recibo {} bytes".format(len(file_bytes)))
    clases, tiempo = get_prediction(image_bytes=file_bytes)
    json_respuesta = {
        'clases': clases,
        'tiempo': tiempo
    }
    log_print("responder: {}".format(json_respuesta))
    return jsonify(json_respuesta)


if __name__ == "__main__":
    app.run(port=7002)
