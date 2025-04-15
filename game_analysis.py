import os
import json
import openai
from typing import List, Dict, Any

class GameAnalyzer:
    """Analyzes game data using OpenAI's API"""
    
    def __init__(self, api_key=None):
        """Initialize with optional API key"""
        if api_key:
            self.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Try to get from environment
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
    
    def analyze_rps_game(self, game_data):
        """Analyze a Rock Paper Scissors game and provide insights"""
        if not self.api_key:
            return "API key not set. Please configure your OpenAI API key."
        
        # Format the game data for analysis
        formatted_game = self._format_rps_game(game_data)
        
        # Construct the prompt
        prompt = f"""
        Analyze this Rock Paper Scissors game:
        
        {formatted_game}
        
        Provide a brief analysis including:
        1. Patterns in player choices. This should be deeper than just looking at the stats. Try to pick out any patterns given the full context of the game such as "the player consistently switched after losing a round" or "the player always stays with the same movewhen losing a round"
        3. How could the player improve their strategy
        4. Was there any psychological pattern evident in the choices?
        """
        
        try:
            # Call OpenAI API with new format
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a Rock Paper Scissors game analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _format_rps_game(self, game_data):
        """Format Rock Paper Scissors game data into a readable string for analysis"""
        formatted_game = f"Game ID: {game_data['id']}\n"
        formatted_game += f"Outcome: {game_data['outcome'] or 'Unknown'}, Winner: {game_data['winner'] or 'None'}\n\n"
        
        rounds = []
        for move in game_data['moves']:
            if 'move_data' in move and move['move_data']:
                move_data = move['move_data']
                player_choice = move_data.get('player_choice')
                ai_choice = move_data.get('ai_choice')
                result = move_data.get('result')
                
                if player_choice and ai_choice:
                    rounds.append({
                        'player_choice': player_choice,
                        'ai_choice': ai_choice,
                        'result': result
                    })
        
        formatted_game += "Rounds:\n"
        for i, round_data in enumerate(rounds, 1):
            formatted_game += f"Round {i}: Player chose {round_data['player_choice']}, AI chose {round_data['ai_choice']}, "
            formatted_game += f"Result: {round_data['result']}\n"
        
        return formatted_game

    def format_games_for_analysis(self, games: List[Dict[str, Any]]) -> str:
        """Format game data into a readable string for the LLM prompt"""
        formatted_games = []
        for game in games:
            game_str = f"\nGame #{game['id']}:\n"
            game_str += f"Outcome: {'Win for ' + game['winner'] if game['winner'] else 'Tie'}\n"
            game_str += "Moves:\n"
            
            for i, move in enumerate(game['moves'], 1):
                board_str = '\n'.join([' '.join(row) for row in move['board_state']])
                game_str += f"\nMove {i} by {move['player']} at position {move['position']}:\n{board_str}\n"
            
            formatted_games.append(game_str)
        
        return '\n'.join(formatted_games)

    def create_analysis_prompt(self, games_data: List[Dict[str, Any]], question: str) -> str:
        """Create a detailed prompt for the LLM analysis."""
        formatted_games = self.format_games_for_analysis(games_data)
        
        return f"""You are an expert Tic Tac Toe game analyst. Your task is to analyze the provided games and answer the user's question.
        
Game Data:
{formatted_games}

User's Question: {question}

Please provide a detailed analysis that includes:
1. Specific observations about the gameplay patterns
2. Strategic strengths and weaknesses identified
3. Concrete recommendations for improvement
4. References to specific moves or situations from the games

Focus on actionable insights that can help improve the player's strategy."""

    def analyze_games(self, games_data: List[Dict[str, Any]], question: str) -> str:
        """Analyze the games using OpenAI's API and return the analysis."""
        try:
            prompt = self.create_analysis_prompt(games_data, question)
            
            # Use the new API format with non-async call
            response = self.client.chat.completions.create(
                model="gpt-4",  # You can also use "gpt-3.5-turbo" for a more cost-effective option
                messages=[
                    {"role": "system", "content": "You are an expert Tic Tac Toe game analyst, providing detailed and actionable feedback on gameplay."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error analyzing games: {str(e)}") 