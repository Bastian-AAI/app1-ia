# -*- coding: utf-8 -*- 
from app import app
from utils import get_prediction, log_print
from flask import Flask, jsonify, request


@app.route('/ejercicio2/app1-ia/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_bytes = file.read()
    log_print("recibo {} bytes".format(len(file_bytes)))
    clase_id, clase_nombre = get_prediction(image_bytes=file_bytes)
    json_respuesta = {'clase_id': clase_id, 'clase_nombre': clase_nombre}
    log_print("responder: {}".format(json_respuesta))
    return jsonify(json_respuesta)


if __name__ == "__main__":
    app.run(port=7002)
