import os
from flask import Blueprint, request, jsonify, send_file
from games.rockpaperscissors.rps_model import train_model, load_trained_model
from game_history import GameHistory
from werkzeug.utils import secure_filename
import tempfile

ai_bp = Blueprint('ai', __name__)
game_history = GameHistory()

# Configure upload folder
UPLOAD_FOLDER = 'ai_models'
ALLOWED_EXTENSIONS = {'pt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@ai_bp.route('/api/train-ai', methods=['POST'])
def train_ai():
    try:
        data = request.get_json()
        game_id = data.get('game_id')
        
        if not game_id:
            return jsonify({'error': 'Game ID is required'}), 400
            
        # Get the game data
        game = game_history.get_game(game_id)
        if not game:
            return jsonify({'error': 'Game not found'}), 404
            
        # Train the model
        model_path = train_model(game)
        
        if not model_path:
            return jsonify({'error': 'Failed to train model'}), 500
            
        # Return the model URL
        model_url = f'/api/download-model/{os.path.basename(model_path)}'
        return jsonify({'model_url': model_url})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_bp.route('/api/import-ai', methods=['POST'])
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

@ai_bp.route('/api/download-model/<filename>', methods=['GET'])
def download_model(filename):
    try:
        # Ensure the filename is safe
        if not filename.endswith('.pth'):
            return jsonify({'error': 'Invalid file type'}), 400
            
        # Get the model path
        model_path = os.path.join(tempfile.gettempdir(), filename)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
            
        # Send the file
        return send_file(
            model_path,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 