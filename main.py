from flask import Flask, request, jsonify
from flask.wrappers import Request
from PIL import Image
import os
import logging

from src.defines import DATASETS, MODEL_FORWARD_PARAMETERS
from src.forward import TFMForward
from src.model import SAE
from datetime import datetime

app = Flask(__name__)

# Configuración básica de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])

# Obtener el logger
logger = logging.getLogger(__name__)

@app.route('/status', methods=['GET'])
def status():
    logger.info("Solicitud de estado recibida.")
    return jsonify({
        "status": "API is running correctly!",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def process_page_route(req: Request, executeAI = False):
    logger.info(f"Procesando solicitud {'con' if executeAI else 'sin'} AI para la página.")
    
    try:
        # Extraer los datos del formulario
        page_id = int(req.form.get('pageId'))
        dataset_id = int(req.form.get('dataset'))
        logger.info(f"Datos recibidos - pageId: {page_id}, datasetId: {dataset_id}")

        # Verificar y procesar la imagen
        image_file = req.files.get('page')
        if not image_file:
            logger.error("No se proporcionó imagen en la solicitud.")
            return jsonify({"error": "No image provided"}), 400

        logger.info("Imagen recibida y procesada.")
        img = Image.open(image_file)

        # Simulate AI processing (replace this with actual AI model processing)
        if executeAI:
            logger.info("Iniciando el procesamiento AI.")
            boxes = [
                {"x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]}
                for box in
                TFMForward(
                    dataset_name=DATASETS[dataset_id],
                    image=img,
                    forwardParameters=MODEL_FORWARD_PARAMETERS[dataset_id]
                )   
            ]
            logger.info(f"Procesamiento AI completado con {len(boxes)} cajas detectadas.")
        else:
            boxes = [
                {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.3},
                {"x": 0.3, "y": 0.7, "w": 0.3, "h": 0.2}
            ]
            logger.info("Procesamiento simulado completado con 2 cajas detectadas.")

        # Preparar la respuesta
        response = {
            "pageId": page_id,
            "dataset": dataset_id,
            "boxes": boxes
        }
        logger.info(f"Respuesta preparada para pageId: {page_id}, datasetId: {dataset_id}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error durante el procesamiento de la página: {str(e)}")
        return jsonify({"error": "An error occurred during processing"}), 500

@app.route('/test-page', methods=['POST'])
def test_page_post():
    logger.info("Solicitud POST para /test-page recibida.")
    return process_page_route(
        req=request, 
        executeAI=False
        )

@app.route('/process-page', methods=['POST'])
def process_page():
    logger.info("Solicitud POST para /process-page recibida.")
    return process_page_route(
        req=request, 
        executeAI=True
        )

if __name__ == '__main__':
    logger.info("Iniciando la aplicación Flask.")
    try:
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
    except:
        logger.critical(f"Error crítico al iniciar la aplicación Flask: {str(e)}")
        raise  # Re-lanzar la excepción para que también se registre en la consola
