import os
from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from games.game_factory import GameFactory
from game_history import GameHistory
from games.rockpaperscissors.rps_model import load_trained_model, load_model_from_path

game_routes = Blueprint('game_routes', __name__)

@game_routes.route('/rps')
def play_rps():
    """Render the Rock Paper Scissors game page"""
    return render_template('game.html')

@game_routes.route('/custom-ai')
def play_custom_ai():
    """Render the Rock Paper Scissors game page"""
    return render_template('custom_ai.html')

@game_routes.route('/ai-vs-ai')
def play_ai_vs_ai():
    """Render the AI vs. AI battle page"""
    return render_template('ai-vs-ai.html')

@game_routes.route('/api/new-game', methods=['POST'])
def new_game():
    """Create a new game"""
    data = request.get_json()
    game_type = data.get('game_type', 'rps')  # Default to rps
    
    try:
        # Create game instance
        game = GameFactory.create_game(game_type)
        
        # Initialize game history
        game_history = GameHistory()
        
        # Create game state
        game_state = game.create_game_state(game_history)
        
        return jsonify({
            'game_id': game_history.current_game_id,
            'game_type': game_type,
            'board': game_state['board'],
            'player_symbol': game_state['player_symbol'],
            'current_player': game_state['current_player'],
            'game_over': game_state['game_over'],
            'winner': game_state['winner']
        })
    except Exception as e:
        print(f"Error in new_game: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@game_routes.route('/api/move', methods=['POST'])
def make_move():
    """Make a move in the current game"""
    data = request.get_json()
    game_id = data.get('game_id')
    choice = data.get('choice')
    use_custom_model = data.get('use_custom_model', False)
    
    if not game_id:
        return jsonify({'error': 'Game ID is required'}), 400
    
    try:
        # Get game instance and data
        storage = GameHistory().storage
        print(f"Getting game for ID: {game_id}")
        game, game_data = GameFactory.get_game_by_id(game_id, storage)
        print(f"Game instance: {game}")
        print(f"Game data: {game_data}")
        
        if not game or not game_data:
            return jsonify({'error': 'Game not found'}), 404
        
        # Get game history
        game_history = GameHistory()
        game_history.current_game_id = game_id
        
        # Resume game state
        print("Resuming game state")
        game_state = game.resume_game_state(game_history, game_data)
        print(f"Game state: {game_state}")
        
        # Load custom model if requested
        if use_custom_model:
            try:
                print("\n=== ATTEMPTING TO LOAD CUSTOM MODEL ===")
                model_path = os.path.join('uploads', 'custom_model.pth')
                print(f"Looking for model at: {model_path}")
                if os.path.exists(model_path):
                    print("Model file found, loading...")
                    custom_model = load_model_from_path(model_path)
                    print(f"Model loaded successfully: {custom_model}")
                    game_state['board']['custom_ai_model'] = custom_model
                    print("Custom model added to game state")
                else:
                    print("No custom model file found")
                print("=== END CUSTOM MODEL LOADING ===\n")
            except Exception as e:
                print(f"\n=== ERROR LOADING CUSTOM MODEL ===")
                print(f"Error details: {e}")
                print("Continuing with default AI")
                print("=== END CUSTOM MODEL ERROR ===\n")
                # Continue with default AI if model loading fails
        
        # Make player move
        move_data = {'choice': choice}
        game_state, success = game.make_player_move(game_state, move_data)
        
        if not success:
            return jsonify({'error': 'Invalid move'}), 400
        
        # Create a clean board state without the model for storage
        clean_board = game_state['board'].copy()
        if 'custom_ai_model' in clean_board:
            del clean_board['custom_ai_model']
        
        # Save the updated game state
        game_history.record_move(
            game_state['player_symbol'],
            None,
            clean_board,
            move_data
        )
        
        # Create another clean board for the response
        response_board = clean_board.copy()
        
        # Return the updated game state
        return jsonify({
            'board': {
                'player_choice': response_board['player_choice'],
                'ai_choice': response_board['ai_choice'],
                'round': response_board['round'],
                'player_score': response_board['player_score'],
                'ai_score': response_board['ai_score'],
                'draws': response_board['draws'],
                'result': response_board['result'],
                'using_custom_model': bool(game_state['board'].get('custom_ai_model'))
            },
            'game_over': game_state['game_over'],
            'winner': game_state['winner'],
            'current_player': game_state['current_player']
        })
    except Exception as e:
        print(f"Error in make_move: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@game_routes.route('/api/end-game/<game_id>', methods=['POST'])
def end_game(game_id):
    """End the current game and redirect to analysis"""
    try:
        # Get game instance
        storage = GameHistory().storage
        game, game_data = GameFactory.get_game_by_id(game_id, storage)
        
        if not game or not game_data:
            return jsonify({'error': 'Game not found'}), 404
        
        # Create game history with the existing game ID
        game_history = GameHistory()
        game_history.current_game_id = game_id
        
        # End the game in history
        player_score = game_data['moves'][-1]['board_state']['player_score'] if game_data['moves'] else 0
        ai_score = game_data['moves'][-1]['board_state']['ai_score'] if game_data['moves'] else 0
        
        if player_score > ai_score:
            winner = "Player"
        elif ai_score > player_score:
            winner = "AI"
        else:
            winner = None  # Tie
        
        game_history.end_game('completed', winner)
        
        # Return redirect URL to the new analysis page
        return jsonify({
            'redirect_url': f'/analyze-game/{game_id}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@game_routes.route('/api/upload-model', methods=['POST'])
def upload_model():
    """Handle model file upload"""
    if 'model' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['model']
    if not file.filename.endswith('.pth'):
        return jsonify({'error': 'Only .pth files are accepted'}), 400
        
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        
        # Save the file
        file_path = os.path.join('uploads', 'custom_model.pth')
        file.save(file_path)
        
        return jsonify({'message': 'Model uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 