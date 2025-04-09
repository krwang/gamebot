import random
from games.game_base import Game
from game_history import GameHistory

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
            "result": None
        }
    
    def create_game_state(self, game_history, player_symbol=None):
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
            'game_type': self.get_game_type(),
            'max_rounds': 5  # Best of 5 rounds
        }
        
        return game_state
    
    def resume_game_state(self, game_history, game_data):
        """Resume a Rock Paper Scissors game from stored data"""
        # Get the last state of the game
        last_move = game_data['moves'][-1] if game_data['moves'] else None
        board_state = last_move['board_state'] if last_move else self.initialize_game()
        
        game_state = {
            'board': board_state,
            'player_symbol': game_data['player_symbol'],
            'ai_symbol': game_data['ai_symbol'],
            'current_player': "Player",  # Always player's turn in RPS
            'game_over': False,
            'winner': None,
            'history': game_history,
            'game_type': self.get_game_type(),
            'max_rounds': 5  # Best of 5 rounds
        }
        
        return game_state
    
    def check_game_over(self, game_state):
        """Check if the game is over (reached max rounds or player/AI has majority wins)"""
        board = game_state['board']
        player_score = board['player_score']
        ai_score = board['ai_score']
        current_round = board['round']
        max_rounds = game_state['max_rounds']
        
        # Check if either player has a majority of wins
        majority_threshold = (max_rounds // 2) + 1
        
        if player_score >= majority_threshold:
            return True, game_state['player_symbol']
        
        if ai_score >= majority_threshold:
            return True, game_state['ai_symbol']
        
        # Check if we've played all rounds
        if current_round > max_rounds:
            # Determine winner based on score
            if player_score > ai_score:
                return True, game_state['player_symbol']
            elif ai_score > player_score:
                return True, game_state['ai_symbol']
            else:
                return True, None  # Tie
        
        # Game still in progress
        return False, None
    
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
        
        game_state['history'].record_move(
            game_state['player_symbol'],
            None,  # No position for RPS
            game_state['board'],
            move_data
        )
        
        # Prepare for next round
        game_state['board']['round'] += 1
        game_state['board']['player_choice'] = None
        game_state['board']['ai_choice'] = None
        game_state['board']['result'] = None
        
        # Check if game is over
        is_over, winner = self.check_game_over(game_state)
        if is_over:
            game_state['game_over'] = True
            game_state['winner'] = winner
            
            if winner:
                game_state['history'].end_game('win', winner)
            else:
                game_state['history'].end_game('tie')
        
        return game_state, True
    
    def _make_ai_choice(self, game_state):
        """AI makes a random choice"""
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
    
    def make_ai_move(self, game_state):
        """AI moves are handled within make_player_move for this game"""
        # In Rock Paper Scissors, AI makes its move right after the player
        # This is already handled in make_player_move
        return game_state, False 