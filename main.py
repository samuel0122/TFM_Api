from flask import Flask, request, jsonify, abort, send_file
from PIL import Image
import os
import logging
from datetime import datetime

from src.defines import DATASETS, MODEL_FORWARD_PARAMETERS
from src.forward import TFMForward
from src.model import SAE
from src.utils import get_dataset_folder, get_mimetype


app = Flask(__name__)

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Rutas de la API
API_VERSION = '/api/v1'
API_GET_STATUS = f'{API_VERSION}/status'
API_POST_PROCESS_IMAGE = f'{API_VERSION}/process-image'
API_GET_IMAGES = f'{API_VERSION}/images/<dataset>'
API_GET_IMAGE = f'{API_VERSION}/image/<dataset>/<filename>'

### Rutas de la API ###

# Obtener estado

@app.route(API_GET_STATUS, methods=['GET'])
def status():
    """Devuelve el estado de la API"""
    logger.info("Solicitud de estado recibida.")
    return jsonify({
        "status": "API is running correctly!",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# Procesar imagen

@app.route(API_POST_PROCESS_IMAGE, methods=['POST'])
def process_page():
    """Procesa una imagen y devuelve las cajas detectadas"""
    logger.info(f"Procesando solicitud con AI para la página.")
    
    try:
        page_id = int(request.form.get('pageId'))
        dataset_id = int(request.form.get('dataset'))
        logger.info(f"Datos recibidos - pageId: {page_id}, datasetId: {dataset_id}")

        image_file = request.files.get('page')
        if not image_file:
            logger.error("No se proporcionó imagen en la solicitud.")
            return jsonify({"error": "No image provided"}), 400

        logger.info("Imagen recibida y procesada.")
        img = Image.open(image_file)

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

# Obtener imagenes

@app.route(API_GET_IMAGES, methods=['GET'])
def list_images(dataset):
    """Lista todas las imágenes de un dataset específico"""
    folder = get_dataset_folder(dataset)
    
    if not folder:
        abort(404, description="Dataset not found")
        
    try:
        files = os.listdir(folder)
    except FileNotFoundError:
        abort(404, description="Dataset folder not found")
    
    images = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        abort(404, description="No images found in dataset")

    return jsonify(images)

# Obtener imagen específico

@app.route(API_GET_IMAGE, methods=['GET'])
def get_image(dataset, filename):
    """Devuelve una imagen de un dataset específico"""
    folder = get_dataset_folder(dataset)
    
    if not folder:
        abort(404, description="Dataset not found")
    
    image_path = os.path.join(folder, filename)

    if not os.path.exists(image_path):
        abort(404, description="Image not found")

    mimetype = get_mimetype(filename)
    
    if not mimetype:
        abort(415, description="Unsupported media type")

    try:
        return send_file(image_path, mimetype=mimetype)
    except Exception as e:
        logger.error(f"Error al servir la imagen: {str(e)}")
        abort(500, description=f"Error serving image: {str(e)}")

    
if __name__ == '__main__':
    logger.info("Iniciando la aplicación Flask.")
    try:
        port = int(os.environ.get("PORT", 5001))
        app.run(debug=True, host='0.0.0.0', port=port)
    except Exception as e:
        logger.critical(f"Error crítico al iniciar la aplicación Flask: {str(e)}")
        raise
