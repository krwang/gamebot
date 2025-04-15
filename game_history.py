from game_storage import GameStorage

class GameHistory:
    def __init__(self):
        """Initialize game history with storage backend"""
        self.storage = GameStorage()
        self.current_game_id = None
        self.game_type = None
    
    def start_new_game(self, game_type, player_symbol):
        """Start tracking a new game"""
        self.game_type = game_type
        self.current_game_id = self.storage.start_game(game_type, player_symbol)
    
    def record_move(self, player, position, board_state, move_data=None):
        """Record a single move with the current board state"""
        if self.current_game_id is None:
            raise RuntimeError("No active game - call start_new_game() first")
        
        self.storage.record_move(
            self.current_game_id,
            player,
            position,
            board_state,
            move_data
        )
    
    def end_game(self, outcome, winner=None):
        """End the current game with given outcome"""
        if self.current_game_id is None:
            raise RuntimeError("No active game - call start_new_game() first")
        
        self.storage.end_game(self.current_game_id, outcome, winner)
        self.current_game_id = None
        self.game_type = None
    
    def get_recent_games(self, game_type=None, limit=10):
        """Get the most recent games in JSON format"""
        return self.storage.get_games_as_json(game_type=game_type, limit=limit)
    
    def get_games_by_outcome(self, outcome, game_type=None):
        """Get all games with specific outcome (win/tie)"""
        return self.storage.get_games_by_outcome(outcome, game_type=game_type)
    
    def get_game(self, game_id):
        """Get a specific game by ID"""
        return self.storage.get_game(game_id)
    
    def close(self):
        """Close the storage connection"""
        self.storage.close() 