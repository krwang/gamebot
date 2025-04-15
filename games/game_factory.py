from games.rockpaperscissors.game import RockPaperScissorsGame

class GameFactory:
    """Factory class to create and manage game instances"""
    
    # Dictionary of game types and their respective classes
    _game_types = {
        "rockpaperscissors": RockPaperScissorsGame
    }
    
    @classmethod
    def get_game_types(cls):
        """Return a list of available game types"""
        return list(cls._game_types.keys())
    
    @classmethod
    def create_game(cls, game_type):
        """Create and return an instance of the specified game type"""
        if game_type not in cls._game_types:
            raise ValueError(f"Unknown game type: {game_type}")
        
        return cls._game_types[game_type]()
    
    @classmethod
    def get_game_by_id(cls, game_id, storage):
        """Get a game instance based on the game_id"""
        # Get the game data from storage
        game_data = storage.get_game_as_json(game_id)
        if not game_data:
            return None, None
        
        # Get the game type
        game_type = game_data["game_type"]
        
        # Create an instance of the appropriate game class
        game_instance = cls.create_game(game_type)
        
        return game_instance, game_data
    
    @classmethod
    def resume_game_state(cls, game_id, storage):
        """Resume a game with its state from storage"""
        # Get the game data from storage
        game_data = storage.get_game(game_id)
        if not game_data:
            return None
        
        # Get the game type
        game_type = game_data["game_type"]
        
        # Create an instance of the appropriate game class
        game_instance = cls.create_game(game_type)
        
        # Load the game state based on game type            
        if game_type == "rockpaperscissors":
            # For Rock Paper Scissors, load the game state
            game_instance.rounds = game_data.get("rounds", [])
            game_instance.player_score = game_data.get("player_score", 0)
            game_instance.computer_score = game_data.get("computer_score", 0)
            game_instance.round_number = game_data.get("round_number", 1)
            # Don't include the entire game_data in the returned game object
            return game_instance
        
        return game_instance 