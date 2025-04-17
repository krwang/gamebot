from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from games.game_factory import GameFactory
from game_history import GameHistory

game_routes = Blueprint('game_routes', __name__)

@game_routes.route('/rps')
def play_rps():
    """Render the Rock Paper Scissors game page"""
    return render_template('game.html')

@game_routes.route('/custom-ai')
def play_custom_ai():
    """Render the Rock Paper Scissors game page"""
    return render_template('custom_ai.html')

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
        
        # Make player move
        move_data = {'choice': choice}
        game_state, success = game.make_player_move(game_state, move_data)
        
        if not success:
            return jsonify({'error': 'Invalid move'}), 400
        
        # Save the updated game state
        game_history.record_move(
            game_state['player_symbol'],
            None,
            game_state['board'],
            move_data
        )
        
        # Return the updated game state
        return jsonify({
            'board': {
                'player_choice': game_state['board']['player_choice'],
                'ai_choice': game_state['board']['ai_choice'],
                'round': game_state['board']['round'],
                'player_score': game_state['board']['player_score'],
                'ai_score': game_state['board']['ai_score'],
                'draws': game_state['board']['draws'],
                'result': game_state['board']['result']
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