import random
from games.game_base import Game
from game_history import GameHistory

class TicTacToeGame(Game):
    def get_game_type(self):
        return "tictactoe"
    
    def initialize_game(self):
        """Initialize a new TicTacToe game state"""
        return [[" " for _ in range(3)] for _ in range(3)]
    
    def create_game_state(self, game_history, player_symbol=None):
        """Create a new game state for TicTacToe"""
        board = self.initialize_game()
        
        # Randomly assign X or O to the player if not specified
        if player_symbol is None:
            player_symbol = random.choice(['X', 'O'])
            
        ai_symbol = 'O' if player_symbol == 'X' else 'X'
        
        # Start new game with player symbol and game type
        game_history.start_new_game(self.get_game_type(), player_symbol)
        
        game_state = {
            'board': board,
            'player_symbol': player_symbol,
            'ai_symbol': ai_symbol,
            'current_player': 'X',  # X always goes first
            'game_over': False,
            'winner': None,
            'history': game_history,
            'game_type': self.get_game_type()
        }
        
        return game_state
    
    def resume_game_state(self, game_history, game_data):
        """Resume a TicTacToe game from stored data"""
        last_move = game_data['moves'][-1] if game_data['moves'] else None
        board = last_move['board_state'] if last_move else self.initialize_game()
        
        # Determine player symbols from the stored data
        player_symbol = game_data['player_symbol']
        ai_symbol = game_data['ai_symbol']
        
        # Determine whose turn it is
        moves = game_data['moves']
        current_player = 'X' if len(moves) % 2 == 0 else 'O'
        
        game_state = {
            'board': board,
            'player_symbol': player_symbol,
            'ai_symbol': ai_symbol,
            'current_player': current_player,
            'game_over': False,
            'winner': None,
            'history': game_history,
            'game_type': self.get_game_type()
        }
        
        return game_state
    
    def check_winner(self, board, player):
        """Check if the given player has won"""
        win_conditions = [
            [board[0][0], board[0][1], board[0][2]],
            [board[1][0], board[1][1], board[1][2]],
            [board[2][0], board[2][1], board[2][2]],
            [board[0][0], board[1][0], board[2][0]],
            [board[0][1], board[1][1], board[2][1]],
            [board[0][2], board[1][2], board[2][2]],
            [board[0][0], board[1][1], board[2][2]],
            [board[0][2], board[1][1], board[2][0]]
        ]
        return [player, player, player] in win_conditions
    
    def get_empty_positions(self, board):
        """Get all empty positions on the board"""
        return [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]
    
    def check_game_over(self, game_state):
        """Check if the game is over and return (is_over, winner)"""
        board = game_state['board']
        
        # Check if either player has won
        if self.check_winner(board, game_state['player_symbol']):
            return True, game_state['player_symbol']
        
        if self.check_winner(board, game_state['ai_symbol']):
            return True, game_state['ai_symbol']
        
        # Check for a tie (no empty spaces left)
        if not self.get_empty_positions(board):
            return True, None
        
        # Game is still in progress
        return False, None
    
    def make_player_move(self, game_state, move_data):
        """Make a player move and update the game state"""
        if game_state['game_over']:
            return game_state, False
        
        if game_state['current_player'] != game_state['player_symbol']:
            return game_state, False
        
        row, col = move_data['row'], move_data['col']
        
        # Check if the move is valid
        if game_state['board'][row][col] != " ":
            return game_state, False
        
        # Make the move
        game_state['board'][row][col] = game_state['player_symbol']
        game_state['history'].record_move(
            game_state['player_symbol'], 
            (row, col), 
            game_state['board']
        )
        
        # Check if game is over after player's move
        is_over, winner = self.check_game_over(game_state)
        if is_over:
            game_state['game_over'] = True
            game_state['winner'] = winner
            
            if winner:
                game_state['history'].end_game('win', winner)
            else:
                game_state['history'].end_game('tie')
                
            return game_state, True
        
        # Switch to AI's turn
        game_state['current_player'] = game_state['ai_symbol']
        return game_state, True
    
    def make_ai_move(self, game_state):
        """Make an AI move and update the game state"""
        if game_state['game_over']:
            return game_state, False
        
        if game_state['current_player'] != game_state['ai_symbol']:
            return game_state, False
        
        # Get all empty positions
        empty_positions = self.get_empty_positions(game_state['board'])
        if not empty_positions:
            return game_state, False
        
        # Choose a random empty position
        ai_row, ai_col = random.choice(empty_positions)
        
        # Make the move
        game_state['board'][ai_row][ai_col] = game_state['ai_symbol']
        game_state['history'].record_move(
            game_state['ai_symbol'], 
            (ai_row, ai_col), 
            game_state['board']
        )
        
        # Check if game is over after AI's move
        is_over, winner = self.check_game_over(game_state)
        if is_over:
            game_state['game_over'] = True
            game_state['winner'] = winner
            
            if winner:
                game_state['history'].end_game('win', winner)
            else:
                game_state['history'].end_game('tie')
                
            return game_state, True
        
        # Switch back to player's turn
        game_state['current_player'] = game_state['player_symbol']
        return game_state, True 