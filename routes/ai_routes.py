import os
import logging
from flask import Blueprint, request, jsonify, send_file
from games.rockpaperscissors.rps_model import train_model, load_trained_model
from game_history import GameHistory
from werkzeug.utils import secure_filename
import tempfile
import boto3

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ai_bp = Blueprint('ai', __name__)
game_history = GameHistory()

# Configure upload folder
UPLOAD_FOLDER = 'ai_models'
ALLOWED_EXTENSIONS = {'pt'}

# Configure S3
S3_BUCKET = 'rps-ai'
s3_client = boto3.client('s3')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@ai_bp.route('/api/train-ai', methods=['POST'])
def train_ai():
    try:
        logger.info("Received train-ai request")
        data = request.get_json()
        logger.debug(f"Request data: {data}")
        
        game_id = data.get('game_id')
        upload_to_s3 = data.get('upload_to_s3', False)
        model_name = data.get('model_name')
        
        if not game_id:
            logger.error("No game_id provided in request")
            return jsonify({'error': 'Game ID is required'}), 400
            
        if not model_name:
            logger.error("No model_name provided in request")
            return jsonify({'error': 'Model name is required'}), 400
            
        logger.info(f"Attempting to get game data for game_id: {game_id}")
        # Get the game data
        game = game_history.get_game(game_id)
        if not game:
            logger.error(f"Game not found for game_id: {game_id}")
            return jsonify({'error': 'Game not found'}), 404
            
        logger.info("Game data retrieved successfully")
        logger.debug(f"Game data keys: {game.keys() if isinstance(game, dict) else 'Not a dictionary'}")
        if isinstance(game, dict) and 'moves' in game:
            logger.debug(f"Number of moves: {len(game['moves'])}")
            logger.debug(f"First move structure: {game['moves'][0] if game['moves'] else 'No moves'}")
            
        logger.info("Starting model training")
        # Train the model
        model_path = train_model(game)
        
        if not model_path:
            logger.error("Model training failed - no model path returned")
            return jsonify({'error': 'Failed to train model'}), 500
            
        logger.info(f"Model trained successfully, saved to: {model_path}")
        
        if upload_to_s3:
            try:
                # Upload to S3 with custom model name
                s3_key = f"models/{model_name}.pth"
                s3_client.upload_file(model_path, S3_BUCKET, s3_key)
                logger.info(f"Model uploaded to S3: {s3_key}")
                
                # Clean up local file
                os.remove(model_path)
                
                return jsonify({
                    'message': 'Model trained and uploaded to S3 successfully',
                    's3_key': s3_key
                })
            except Exception as e:
                logger.error(f"Failed to upload model to S3: {str(e)}")
                return jsonify({'error': f'Failed to upload model to S3: {str(e)}'}), 500
        else:
            # Return the model URL for download
            model_url = f'/api/download-model/{os.path.basename(model_path)}'
            logger.info(f"Returning model URL: {model_url}")
            return jsonify({'model_url': model_url})
        
    except Exception as e:
        logger.exception("Unexpected error in train_ai route")
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

@ai_bp.route('/api/load-model', methods=['GET'])
def load_model():
    try:
        model_name = request.args.get('model')
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400

        # Construct the S3 key
        s3_key = f"models/{model_name}.pth"
        
        try:
            # Download the model from S3
            local_path = os.path.join(tempfile.gettempdir(), f"{model_name}.pth")
            s3_client.download_file(S3_BUCKET, s3_key, local_path)
            
            return jsonify({
                'message': 'Model loaded successfully',
                'model_path': local_path
            })
        except Exception as e:
            logger.error(f"Failed to load model from S3: {str(e)}")
            return jsonify({'error': f'Failed to load model from S3: {str(e)}'}), 500
            
    except Exception as e:
        logger.exception("Unexpected error in load_model route")
        return jsonify({'error': str(e)}), 500

@ai_bp.route('/api/list-models', methods=['GET'])
def list_models():
    """List all available models in the S3 bucket"""
    try:
        # List objects in the models directory
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='models/'
        )
        
        # Extract model names from the S3 keys
        models = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.pth'):
                    # Remove 'models/' prefix and '.pth' suffix
                    model_name = obj['Key'][7:-4]
                    models.append(model_name)
        
        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"Failed to list models from S3: {str(e)}")
        return jsonify({'error': str(e)}), 500
 