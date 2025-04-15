import random
import os
import torch
from games.game_base import Game
from game_history import GameHistory
from .rps_model import load_trained_model, predict_next_move, train_model

class RockPaperScissorsGame(Game):
    # Define the valid choices
    CHOICES = ["rock", "paper", "scissors"]
    
    # Define the rules: winner[choice1][choice2] returns who wins
    RULES = {
        "rock": {
            "rock": "tie",
            "paper": "player2",
            "scissors": "player1"
        },
        "paper": {
            "rock": "player1",
            "paper": "tie",
            "scissors": "player2"
        },
        "scissors": {
            "rock": "player2",
            "paper": "player1",
            "scissors": "tie"
        }
    }
    
    def get_game_type(self):
        return "rockpaperscissors"
    
    def initialize_game(self):
        """Initialize a new Rock Paper Scissors game state"""
        return {
            "player_choice": None,
            "ai_choice": None,
            "round": 1,
            "player_score": 0,
            "ai_score": 0,
            "draws": 0,
            "result": None,
            "custom_ai_model": None
        }
    
    def create_game_state(self, game_history, player_symbol=None, custom_ai_model=None):
        """Create a new game state for Rock Paper Scissors"""
        # Start new game with game type (player_symbol isn't relevant here but we'll use "Player")
        game_history.start_new_game(self.get_game_type(), "Player")
        
        game_state = {
            'board': self.initialize_game(),
            'player_symbol': "Player",
            'ai_symbol': "AI",
            'current_player': "Player",  # Player always goes first
            'game_over': False,
            'winner': None,
            'history': game_history,
            'game_type': self.get_game_type()
        }
        
        # Set custom AI model if provided
        game_state['board']['custom_ai_model'] = custom_ai_model
        
        return game_state
    
    def resume_game_state(self, game_history, game_data):
        """Resume a Rock Paper Scissors game from stored data"""
        # Get the last state of the game
        last_move = game_data['moves'][-1] if game_data['moves'] else None
        
        # Initialize a new board state
        board_state = self.initialize_game()
        
        # If there was a previous move, update the board state with its values
        if last_move and 'board_state' in last_move:
            prev_state = last_move['board_state']
            board_state.update({
                'player_score': prev_state.get('player_score', 0),
                'ai_score': prev_state.get('ai_score', 0),
                'draws': prev_state.get('draws', 0),
                'round': prev_state.get('round', 1)
            })
        
        game_state = {
            'board': board_state,
            'player_symbol': game_data['player_symbol'],
            'ai_symbol': game_data['ai_symbol'],
            'current_player': "Player",  # Always player's turn in RPS
            'game_over': False,
            'winner': None,
            'history': game_history,
            'game_type': self.get_game_type()
        }
        
        return game_state
    
    def check_game_over(self, game_state):
        """Check if the game is over (only when explicitly ended)"""
        # Game only ends when explicitly ended by the player
        return game_state['game_over'], game_state['winner']
    
    def make_player_move(self, game_state, move_data):
        """Process player's choice in Rock Paper Scissors"""
        if game_state['game_over']:
            return game_state, False
        
        # Get player's choice
        player_choice = move_data.get('choice')
        if player_choice not in self.CHOICES:
            return game_state, False
        
        # Record player's choice
        game_state['board']['player_choice'] = player_choice
        
        # AI makes choice immediately after player
        self._make_ai_choice(game_state)
        
        # Determine round result
        self._determine_round_result(game_state)
        
        # Record the move with both player and AI choices
        move_data = {
            'player_choice': game_state['board']['player_choice'],
            'ai_choice': game_state['board']['ai_choice'],
            'result': game_state['board']['result']
        }
        
        # Increment round before recording the move
        game_state['board']['round'] += 1
        
        game_state['history'].record_move(
            game_state['player_symbol'],
            None,  # No position for RPS
            game_state['board'],
            move_data
        )
        
        return game_state, True
    
    def _make_ai_choice(self, game_state):
        """AI makes a choice using the custom model if available, otherwise random"""
        board = game_state['board']
        
        # For now, just use random choice
        ai_choice = random.choice(self.CHOICES)
        game_state['board']['ai_choice'] = ai_choice
    
    def _determine_round_result(self, game_state):
        """Determine the result of a round and update scores"""
        player_choice = game_state['board']['player_choice']
        ai_choice = game_state['board']['ai_choice']
        
        result = self.RULES[player_choice][ai_choice]
        
        if result == "player1":
            # Player wins
            game_state['board']['result'] = "player_win"
            game_state['board']['player_score'] += 1
        elif result == "player2":
            # AI wins
            game_state['board']['result'] = "ai_win"
            game_state['board']['ai_score'] += 1
        else:
            # Tie
            game_state['board']['result'] = "tie"
            game_state['board']['draws'] = game_state['board'].get('draws', 0) + 1
    
    def make_ai_move(self, game_state):
        """AI moves are handled within make_player_move for this game"""
        # In Rock Paper Scissors, AI makes its move right after the player
        # This is already handled in make_player_move
        return game_state, False
    
    def train_custom_ai(self, game_id, model_path):
        """Train a custom AI model based on a specific game"""
        storage = GameStorage()
        game_data = storage.get_game_as_json(game_id)
        
        if not game_data:
            return False, "Game not found"
        
        try:
            # Train the model
            model = train_model([game_data])
            
            # Save the model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model, model_path)
            
            return True, model_path
        except Exception as e:
            return False, str(e) 