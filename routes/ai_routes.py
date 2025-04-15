import os
from flask import Blueprint, request, jsonify, send_file
from games.rockpaperscissors.game import RockPaperScissorsGame
from werkzeug.utils import secure_filename

ai_routes = Blueprint('ai_routes', __name__)

# Configure upload folder
UPLOAD_FOLDER = 'ai_models'
ALLOWED_EXTENSIONS = {'pt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@ai_routes.route('/api/train-ai', methods=['POST'])
def train_ai():
    """Train a new AI model based on a specific game"""
    data = request.get_json()
    game_id = data.get('game_id')
    
    if not game_id:
        return jsonify({'error': 'Game ID is required'}), 400
    
    # Create model path
    model_filename = f'rps_model_{game_id}.pt'
    model_path = os.path.join(UPLOAD_FOLDER, model_filename)
    
    # Train the model
    game = RockPaperScissorsGame()
    success, result = game.train_custom_ai(game_id, model_path)
    
    if success:
        return jsonify({
            'message': 'Model trained successfully',
            'model_url': f'/api/download-model/{model_filename}'
        })
    else:
        return jsonify({'error': result}), 500

@ai_routes.route('/api/import-ai', methods=['POST'])
def import_ai():
    """Import a trained AI model"""
    if 'model' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['model']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        
        return jsonify({
            'message': 'Model imported successfully',
            'model_id': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_routes.route('/api/download-model/<filename>', methods=['GET'])
def download_model(filename):
    """Download a trained AI model"""
    try:
        return send_file(
            os.path.join(UPLOAD_FOLDER, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 