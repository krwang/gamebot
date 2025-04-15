from flask import Blueprint, jsonify, request

# Import game-related modules
from game_history import GameHistory
from game_storage import GameStorage
from games.game_factory import GameFactory

# Create Blueprint for game routes
games_bp = Blueprint('games', __name__)

# Initialize services
storage = GameStorage()
games = {}  # Store active games in memory

@games_bp.route('/api/game-types')
def get_game_types():
    """Return the available game types"""
    return jsonify({
        "game_types": GameFactory.get_game_types()
    })

@games_bp.route('/api/new-game', methods=['POST'])
def new_game():
    try:
        data = request.json
        game_type = data.get('game_type', 'rps')  # Default to rps
        print(f"Starting new game of type: {game_type}")
        
        # Create game instance
        game_instance = GameFactory.create_game(game_type)
        game_history = GameHistory()
        
        # Create game state
        game_state = game_instance.create_game_state(game_history)
        
        # Store the game in memory
        game_id = game_history.current_game_id
        games[game_id] = game_state
        
        print(f"New game created with ID: {game_id}, player symbol: {game_state['player_symbol']}, AI symbol: {game_state['ai_symbol']}")
        print(f"Current player: {game_state['current_player']}, is player's turn: {game_state['current_player'] == game_state['player_symbol']}")
        
        return jsonify({
            'game_id': game_id,
            'game_type': game_type,
            'board': game_state['board'],
            'player_symbol': game_state['player_symbol'],
            'current_player': game_state['current_player'],
            'game_over': game_state['game_over'],
            'winner': game_state['winner'],
            'is_player_turn': game_state['current_player'] == game_state['player_symbol']
        })
    except Exception as e:
        print(f"Error in new_game: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@games_bp.route('/api/recent-games')
def get_recent_games():
    try:
        limit = request.args.get('limit', 10, type=int)
        game_type = request.args.get('game_type')
        recent_games = storage.get_games_as_json(game_type=game_type, limit=limit)
        return jsonify({'games': recent_games})
    except Exception as e:
        print(f"Error in get_recent_games: {str(e)}")
        return jsonify({'error': str(e)}), 500

@games_bp.route('/api/game/<game_id>')
def get_game(game_id):
    try:
        game = storage.get_game(game_id)
        if not game:
            return jsonify({'error': 'Game not found'}), 404
        return jsonify({'game': game})
    except Exception as e:
        print(f"Error in get_game: {str(e)}")
        return jsonify({'error': str(e)}), 500 