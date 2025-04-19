import os
import logging
from flask import Blueprint, request, jsonify, send_file, Response
from games.rockpaperscissors.rps_model import train_model, load_trained_model, load_model_from_path, predict_next_move
from game_history import GameHistory
from games.game_factory import GameFactory
from werkzeug.utils import secure_filename
import tempfile
import boto3
import json
import time
import random

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

@ai_bp.route('/api/custom-ai', methods=['POST'])
def custom_ai():
    try:
        data = request.get_json()
        model_name = data.get('model')
        player_choice = data.get('choice')
        
        if not model_name or not player_choice:
            return jsonify({'error': 'Model name and player choice are required'}), 400
            
        logger.info(f"Custom AI: Loading model {model_name} for move against player choice {player_choice}")
            
        # Load the model
        model_path = os.path.join(tempfile.gettempdir(), f"{model_name}.pth")
        if not os.path.exists(model_path):
            logger.info(f"Downloading model {model_name} from S3")
            s3_client.download_file(S3_BUCKET, f"models/{model_name}.pth", model_path)
            
        model = load_model_from_path(model_path)
        
        # Get game history
        game_history = GameHistory()
        history = game_history.get_current_game_history()
        logger.info(f"Current game history length: {len(history)}")
        
        # Get AI move using non-deterministic prediction
        ai_choice = predict_next_move(model, history, deterministic=False)
        logger.info(f"Model {model_name} chose {ai_choice}")
        
        # Determine winner
        if player_choice == ai_choice:
            result = 'tie'
        elif (player_choice == 'rock' and ai_choice == 'scissors') or \
             (player_choice == 'paper' and ai_choice == 'rock') or \
             (player_choice == 'scissors' and ai_choice == 'paper'):
            result = 'player_win'
        else:
            result = 'ai_win'
            
        logger.info(f"Round result: {result}")
            
        # Update game history
        game_history.record_move({
            'player_choice': player_choice,
            'ai_choice': ai_choice,
            'result': result
        })
        
        return jsonify({
            'ai_choice': ai_choice,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error in custom AI: {str(e)}")
        return jsonify({'error': str(e)}), 500

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

@ai_bp.route('/api/ai-battle', methods=['POST'])
def ai_battle():
    """Handle an AI vs. AI battle"""
    try:
        data = request.get_json()
        model1_name = data.get('model1')
        model2_name = data.get('model2')
        rounds = data.get('rounds', 10)
        
        logger.info(f"Starting AI battle between {model1_name} and {model2_name} for {rounds} rounds")
        
        if not model1_name or not model2_name:
            return jsonify({'error': 'Both model names are required'}), 400
            
        # Load both models
        model1_path = os.path.join(tempfile.gettempdir(), f"{model1_name}.pth")
        model2_path = os.path.join(tempfile.gettempdir(), f"{model2_name}.pth")
        
        logger.info(f"Looking for models at: {model1_path} and {model2_path}")
        
        # Download models from S3 if not already present
        if not os.path.exists(model1_path):
            logger.info(f"Downloading model {model1_name} from S3")
            s3_client.download_file(S3_BUCKET, f"models/{model1_name}.pth", model1_path)
        if not os.path.exists(model2_path):
            logger.info(f"Downloading model {model2_name} from S3")
            s3_client.download_file(S3_BUCKET, f"models/{model2_name}.pth", model2_path)
            
        # Load the models
        logger.info("Loading models")
        model1 = load_model_from_path(model1_path)
        model2 = load_model_from_path(model2_path)
        
        logger.info(f"Model 1 type: {type(model1)}")
        logger.info(f"Model 2 type: {type(model2)}")
        
        # Initialize game state
        game = GameFactory.create_game('rockpaperscissors')
        game_history = GameHistory()
        game_state = game.create_game_state(game_history)
        
        # Initialize scores
        model1_score = 0
        model2_score = 0
        draws = 0
        
        # Initialize separate move histories for each model
        model1_history = []
        model2_history = []
        
        # Add initial random moves to each model's history
        choices = ['rock', 'paper', 'scissors']
        for _ in range(5):
            # For model1's history, model1 is the player and model2 is the AI
            model1_choice = random.choice(choices)
            model2_choice = random.choice(choices)
            result = 'tie' if model1_choice == model2_choice else \
                    'player_win' if (model1_choice == 'rock' and model2_choice == 'scissors') or \
                                   (model1_choice == 'paper' and model2_choice == 'rock') or \
                                   (model1_choice == 'scissors' and model2_choice == 'paper') else \
                    'ai_win'
            model1_history.append({
                'move_data': {
                    'player_choice': model1_choice,
                    'ai_choice': model2_choice,
                    'result': result
                },
                'board_state': {
                    'player_score': 0,
                    'ai_score': 0,
                    'round': 0
                }
            })
            
            # For model2's history, model2 is the player and model1 is the AI
            result = 'tie' if model2_choice == model1_choice else \
                    'player_win' if (model2_choice == 'rock' and model1_choice == 'scissors') or \
                                   (model2_choice == 'paper' and model1_choice == 'rock') or \
                                   (model2_choice == 'scissors' and model1_choice == 'paper') else \
                    'ai_win'
            model2_history.append({
                'move_data': {
                    'player_choice': model2_choice,
                    'ai_choice': model1_choice,
                    'result': result
                },
                'board_state': {
                    'player_score': 0,
                    'ai_score': 0,
                    'round': 0
                }
            })
        
        def generate():
            nonlocal model1_score, model2_score, draws
            
            # Play the specified number of rounds
            for round_num in range(1, rounds + 1):
                logger.info(f"Playing round {round_num}")
                
                # Get moves from both models using their respective histories
                logger.info("Getting move from model 1")
                model1_choice, model1_confidence = predict_next_move(model1, model1_history, deterministic=False, return_confidence=True)
                logger.info("Getting move from model 2")
                model2_choice, model2_confidence = predict_next_move(model2, model2_history, deterministic=False, return_confidence=True)
                
                # Add randomness based on confidence
                if random.random() < 0.2 and model1_confidence < 0.8:
                    model1_choice = random.choice(['rock', 'paper', 'scissors'])
                if random.random() < 0.2 and model2_confidence < 0.8:
                    model2_choice = random.choice(['rock', 'paper', 'scissors'])
                
                logger.info(f"Model 1 chose {model1_choice} (confidence: {model1_confidence:.2f}), Model 2 chose {model2_choice} (confidence: {model2_confidence:.2f})")
                
                # Update game state
                game_state['board']['player_choice'] = model1_choice
                game_state['board']['ai_choice'] = model2_choice
                
                # Determine winner
                if model1_choice == model2_choice:
                    result = 'tie'
                    draws += 1
                elif (model1_choice == 'rock' and model2_choice == 'scissors') or \
                     (model1_choice == 'paper' and model2_choice == 'rock') or \
                     (model1_choice == 'scissors' and model2_choice == 'paper'):
                    result = 'model1_win'
                    model1_score += 1
                else:
                    result = 'model2_win'
                    model2_score += 1
                
                logger.info(f"Round {round_num} result: {result}")
                
                # Update each model's history with their perspective
                # For model1's history, model1 is the player and model2 is the AI
                model1_result = 'tie' if model1_choice == model2_choice else \
                              'player_win' if (model1_choice == 'rock' and model2_choice == 'scissors') or \
                                             (model1_choice == 'paper' and model2_choice == 'rock') or \
                                             (model1_choice == 'scissors' and model2_choice == 'paper') else \
                              'ai_win'
                model1_history.append({
                    'move_data': {
                        'player_choice': model1_choice,
                        'ai_choice': model2_choice,
                        'result': model1_result
                    },
                    'board_state': {
                        'player_score': model1_score,
                        'ai_score': model2_score,
                        'round': round_num
                    }
                })
                
                # For model2's history, model2 is the player and model1 is the AI
                model2_result = 'tie' if model2_choice == model1_choice else \
                              'player_win' if (model2_choice == 'rock' and model1_choice == 'scissors') or \
                                             (model2_choice == 'paper' and model1_choice == 'rock') or \
                                             (model2_choice == 'scissors' and model1_choice == 'paper') else \
                              'ai_win'
                model2_history.append({
                    'move_data': {
                        'player_choice': model2_choice,
                        'ai_choice': model1_choice,
                        'result': model2_result
                    },
                    'board_state': {
                        'player_score': model2_score,
                        'ai_score': model1_score,
                        'round': round_num
                    }
                })
                
                # Yield round result
                round_data = {
                    'round': round_num,
                    'model1_choice': model1_choice,
                    'model2_choice': model2_choice,
                    'result': result,
                    'model1_score': model1_score,
                    'model2_score': model2_score,
                    'draws': draws
                }
                logger.info(f"Yielding round data: {round_data}")
                yield json.dumps(round_data) + '\n'
                
                # Add a small delay between rounds
                time.sleep(1)
            
            # End the game
            game_history.end_game('completed', 'Model1' if model1_score > model2_score else 'Model2' if model2_score > model1_score else None)
            
            # Yield final result
            final_data = {
                'final': True,
                'model1_score': model1_score,
                'model2_score': model2_score,
                'draws': draws,
                'total_rounds': rounds,
                'game_id': game_history.current_game_id
            }
            logger.info(f"Yielding final data: {final_data}")
            yield json.dumps(final_data) + '\n'
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        logger.error(f"Error in AI battle: {str(e)}")
        return jsonify({'error': str(e)}), 500
 