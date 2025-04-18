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
        game_type = data.get('game_type', 'tictactoe')  # Default to tictactoe
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

@games_bp.route('/api/ai-move/<game_id>', methods=['POST'])
def ai_move(game_id):
    try:
        try:
            int_game_id = int(game_id)  # Convert game_id to integer
            print(f"AI move requested for game ID: {int_game_id}")
        except ValueError:
            print(f"Invalid game ID format: {game_id}")
            return jsonify({'error': 'Invalid game ID format'}), 400
        
        if int_game_id not in games:
            print(f"Game not found: {int_game_id}")
            return jsonify({'error': 'Game not found'}), 404
        
        game_state = games[int_game_id]
        print(f"Game type: {game_state['game_type']}, Current player: {game_state['current_player']}, AI symbol: {game_state['ai_symbol']}")
        
        game_instance = GameFactory.create_game(game_state['game_type'])
        
        game_state, move_successful = game_instance.make_ai_move(game_state)
        
        if not move_successful:
            print(f"AI move failed for game {int_game_id}")
            return jsonify({'error': 'AI move failed'}), 400
        
        print(f"AI move successful for game {int_game_id}, current player now: {game_state['current_player']}")
        
        response_state = {
            'game_type': game_state['game_type'],
            'board': game_state['board'],
            'game_over': game_state['game_over'],
            'winner': game_state['winner'],
            'current_player': game_state['current_player'],
            'is_player_turn': game_state['current_player'] == game_state['player_symbol']
        }
        
        if game_state['game_over']:
            print(f"Game {int_game_id} over after AI move. Winner: {game_state['winner']}")
            del games[int_game_id]
        
        return jsonify(response_state)
    except Exception as e:
        print(f"Error in ai_move: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@games_bp.route('/api/move', methods=['POST'])
def make_move():
    try:
        data = request.json
        game_id = data['game_id']
        print(f"Player move for game ID: {game_id}")
        
        # Try to convert to integer if it's a string representation of an int
        try:
            if isinstance(game_id, str) and game_id.isdigit():
                int_game_id = int(game_id)
                print(f"Converting string game_id {game_id} to int {int_game_id}")
                game_id = int_game_id
        except (ValueError, TypeError):
            pass
            
        # Debug games dictionary to see what IDs are available
        print(f"Available game IDs in memory: {list(games.keys())}")
        
        if game_id not in games:
            print(f"Game {game_id} not found in memory. Attempting to resume from storage.")
            # Try to resume the game from storage
            active_game = storage.get_game(game_id)
            if active_game:
                # Create game history
                game_history = GameHistory()
                game_history.current_game_id = game_id
                
                # Resume the game state
                game_type = active_game['game_type']
                game_instance = GameFactory.create_game(game_type)
                game_state = game_instance.resume_game_state(game_history, active_game)
                
                # Store in memory for the current request
                games[game_id] = game_state
                print(f"Successfully resumed game {game_id} from storage")
            else:
                print(f"Game not found in storage: {game_id}")
                return jsonify({'error': 'Game not found'}), 404
        
        game_state = games[game_id]
        print(f"Game type: {game_state['game_type']}, Current player: {game_state['current_player']}, Player symbol: {game_state['player_symbol']}")
        
        game_instance = GameFactory.create_game(game_state['game_type'])
        
        # Make the player move
        game_state, move_successful = game_instance.make_player_move(game_state, data)
        
        if not move_successful:
            print(f"Invalid player move for game {game_id}")
            return jsonify({'error': 'Invalid move'}), 400
        
        print(f"Player move successful for game {game_id}, current player now: {game_state['current_player']}")
        
        # Get all moves for the game from storage for RPS games
        if game_state['game_type'] == 'rockpaperscissors':
            try:
                # Get the complete game data with all moves from storage
                game_data = storage.get_game(game_id)
                if game_data and 'moves' in game_data:
                    moves = game_data['moves']
                    print(f"Retrieved {len(moves)} moves for RPS game {game_id}")
                    
                    # Create a clean copy of moves without board_state to avoid circular references
                    clean_moves = []
                    for move in game_data['moves']:
                        clean_move = {
                            'player': move.get('player', ''),
                            'position': move.get('position'),
                            'move_data': move.get('move_data', {}),
                            'timestamp': move.get('timestamp', ''),
                            'is_player_move': move.get('is_player_move', False)
                        }
                        # Remove board_state which likely contains circular references
                        if 'board_state' in clean_move:
                            del clean_move['board_state']
                        clean_moves.append(clean_move)
                    
                    # Add moves to the response for RPS games
                    game_state['board']['moves'] = clean_moves
                else:
                    print(f"No moves found in game data for RPS game {game_id}")
            except Exception as e:
                print(f"Error retrieving moves for RPS game: {str(e)}")
                import traceback
                traceback.print_exc()
        
        response_state = {
            'game_type': game_state['game_type'],
            'board': game_state['board'],
            'game_over': game_state['game_over'],
            'winner': game_state['winner'],
            'current_player': game_state['current_player'],
            'is_player_turn': game_state['current_player'] == game_state['player_symbol']
        }
        
        if game_state['game_over']:
            print(f"Game {game_id} over after player move. Winner: {game_state['winner']}")
            del games[game_id]
        
        return jsonify(response_state)
    except Exception as e:
        print(f"Error in make_move: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@games_bp.route('/api/resume-game/<game_id>', methods=['POST'])
def resume_game(game_id):
    try:
        # Try to find the game by ID
        print(f"Attempting to resume game with ID: {game_id}")
        try:
            # Convert to integer if possible - RPS game IDs might be stored as integers
            int_game_id = int(game_id)
            print(f"Converting string ID {game_id} to integer {int_game_id}")
            game_id = int_game_id  # Use the integer form for the rest of the function
        except ValueError:
            # If not an integer ID, keep as is
            pass

        # Get the active game data directly
        active_game = storage.get_game(game_id)
        
        if not active_game:
            print(f"Game not found with ID: {game_id}")
            return jsonify({'error': 'Game not found'}), 404
            
        # Check if the game is already completed
        if active_game.get('end_time') is not None:
            print(f"Game already completed at: {active_game.get('end_time')}")
            return jsonify({'error': 'Game already completed'}), 400
        
        # Create game history
        game_history = GameHistory()
        game_history.current_game_id = game_id
        
        # Resume the game state with proper handling
        game_type = active_game['game_type']
        game_instance = GameFactory.create_game(game_type)
        game_state = game_instance.resume_game_state(game_history, active_game)
        
        # Store in memory for further moves
        games[game_id] = game_state
            
        # Create the game state based on game type
        if game_type == 'tictactoe':
            # For Tic Tac Toe, use the data from game_state directly
            response_state = {
                'game_id': game_id,
                'game_type': 'tictactoe',
                'board': game_state['board'] if 'board' in game_state else [],
                'player_symbol': game_state.get('player_symbol', ''),
                'current_player': game_state.get('current_player', ''),
                'game_over': game_state.get('game_over', False),
                'winner': game_state.get('winner', None),
                'is_player_turn': game_state.get('current_player', '') == game_state.get('player_symbol', '')
            }
        elif game_type == 'rockpaperscissors':
            # Create a clean board structure without circular references
            board = {}
            
            # Copy required fields from game_state
            if 'board' in game_state and isinstance(game_state['board'], dict):
                # Copy specific fields we need rather than entire board
                for key in ['player_choice', 'ai_choice', 'round', 'player_score', 'ai_score', 'result']:
                    if key in game_state['board']:
                        board[key] = game_state['board'][key]
            
            # Process moves data to avoid circular references
            if 'moves' in active_game:
                # Create a clean copy of moves without board_state to avoid circular references
                clean_moves = []
                for move in active_game['moves']:
                    clean_move = {
                        'player': move.get('player', ''),
                        'position': move.get('position'),
                        'move_data': move.get('move_data', {}),
                        'timestamp': move.get('timestamp', ''),
                        'is_player_move': move.get('is_player_move', False)
                    }
                    # Remove board_state which likely contains circular references
                    if 'board_state' in clean_move:
                        del clean_move['board_state']
                    clean_moves.append(clean_move)
                
                board['moves'] = clean_moves
                print(f"Including {len(clean_moves)} moves in RPS game")
                
                # Debug each move
                for i, move in enumerate(clean_moves):
                    move_data = move.get('move_data', {})
                    if 'player_choice' in move_data:
                        print(f"Move {i+1}: Player: {move_data['player_choice']}, AI: {move_data['ai_choice']}")
            
            response_state = {
                'game_id': game_id,
                'game_type': 'rockpaperscissors',
                'board': board,
                'player_symbol': game_state.get('player_symbol', ''),
                'current_player': game_state.get('current_player', ''),
                'game_over': game_state.get('game_over', False),
                'winner': game_state.get('winner', None),
                'is_player_turn': game_state.get('current_player', '') == game_state.get('player_symbol', '')
            }
        else:
            return jsonify({'error': 'Unknown game type'}), 400
            
        print(f"Prepared game state for resume: {response_state}")
        return jsonify(response_state)
    except Exception as e:
        print(f"Error resuming game: {str(e)}")
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