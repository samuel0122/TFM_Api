from flask import Flask, request, jsonify
from PIL import Image
import io

import model
import defines

app = Flask(__name__)

@app.route('/status', methods=['GET'])
def status():
    from datetime import datetime

    return jsonify({
        "status": "API is running correctly!",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/process-image', methods=['POST'])
def process_image():
    # Get the image from the request
    image_file = request.files.get('image')
    description = request.form.get('description', '')

    if not image_file:
        return jsonify({"error": "No image provided"}), 400

    # Open the image using PIL
    img = Image.open(image_file)

    # Simulate AI processing (replace this with actual AI model processing)
    processed_result = f"Processed image of size {img.size} with description: {description}"

    # Return the result
    return jsonify({"filename": image_file.filename, "result": processed_result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

