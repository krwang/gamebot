from abc import ABC, abstractmethod
from game_history import GameHistory

class Game(ABC):
    """Base abstract class for all games"""
    
    @abstractmethod
    def initialize_game(self):
        """Initialize a new game state"""
        pass
    
    @abstractmethod
    def make_player_move(self, game_state, move_data):
        """Make a player move and return updated game state"""
        pass
    
    @abstractmethod
    def make_ai_move(self, game_state):
        """Make an AI move and return updated game state"""
        pass
    
    @abstractmethod
    def check_game_over(self, game_state):
        """Check if the game is over and return (is_over, winner)"""
        pass
    
    @abstractmethod
    def get_game_type(self):
        """Return a string identifier for this game type"""
        pass
    
    @abstractmethod
    def create_game_state(self, game_history, player_symbol=None):
        """Create and return a new game state with the given GameHistory"""
        pass
    
    @abstractmethod
    def resume_game_state(self, game_history, game_data):
        """Resume a game from stored data"""
        pass 