from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import UAP
import numpy as np


app = Flask(__name__)
CORS(app)


@app.route('/process', methods=['POST'])
def process_route():
    data = request.get_json(force=True)
    img_data = data.get('image')
    if not img_data:
        return jsonify({'error': 'no image provided'}), 400


    # data puede venir como 'data:image/jpeg;base64,xxxx'
    if ',' in img_data:
        header, b64data = img_data.split(',', 1)
    else:
        b64data = img_data
    
    
    try:
        raw = base64.b64decode(b64data)
        img = Image.open(BytesIO(raw)).convert('RGB')
        
        # Convertir a numpy array para OpenCV
        img_np = np.array(img)
    
        # Llamada a tu script
        processed_np = UAP.aplicar_ruido_universal(img_np, intensidad=10, usar_permutacion=False)
    
        # Convertir de vuelta a PIL.Image
        processed = Image.fromarray(processed_np)
    
        # Devolver JPEG
        buf = BytesIO()
        processed.save(buf, format='JPEG', quality=85)
        out_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({'image': 'data:image/jpeg;base64,' + out_b64})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
