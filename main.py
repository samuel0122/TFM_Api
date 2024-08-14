from flask import Flask, request, jsonify
from flask.wrappers import Request
from PIL import Image
import os

from src.defines import DATASETS, MODEL_FORWARD_PARAMETERS
from src.forward import TFMForward
from src.model import SAE
from datetime import datetime

app = Flask(__name__)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "API is running correctly!",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def process_page_route(req: Request, executeAI = False):
    # Extraer los datos del formulario
    page_id = int(req.form.get('pageId'))
    dataset_id = int(req.form.get('dataset'))

    # Verificar y procesar la imagen
    image_file = req.files.get('page')
    if not image_file:
        return jsonify({"error": "No image provided"}), 400

    # Open the image using PIL
    img = Image.open(image_file)

    # Simulate AI processing (replace this with actual AI model processing)
    if executeAI:
        boxes = [
            {"x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]}
            for box in
            TFMForward(
                dataset_name=DATASETS[dataset_id],
                image=img,
                forwardParameters=MODEL_FORWARD_PARAMETERS[dataset_id]
            )   
        ]
        
    else:
        boxes = [
            {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.3},
            {"x": 0.3, "y": 0.7, "w": 0.3, "h": 0.2}
        ]

    # Preparar la respuesta
    response = {
        "pageId": page_id,
        "dataset": dataset_id,
        "boxes": boxes
    }

    return jsonify(response)

@app.route('/test-page', methods=['POST'])
def test_page_post():
    return process_page_route(
        req=request, 
        executeAI=False
        )

@app.route('/process-page', methods=['POST'])
def process_page():
    return process_page_route(
        req=request, 
        executeAI=True
        )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))

