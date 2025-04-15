from flask import Blueprint, jsonify, request, render_template

# Import analyzer and storage
from game_analysis import GameAnalyzer
from game_storage import GameStorage
import os

# Create Blueprint for analysis routes
analysis_bp = Blueprint('analysis', __name__)

# Initialize services
storage = GameStorage()
game_analyzer = GameAnalyzer(api_key=os.getenv('OPENAI_API_KEY'))

@analysis_bp.route('/analyze-game/<game_id>')
def view_game_analysis(game_id):
    """Display the analysis of a single game in a formatted HTML page"""
    try:
        # Get the game data
        game_data = storage.get_game_as_json(game_id)
        if not game_data:
            return render_template('error.html', error='Game not found'), 404
        
        # Check if analysis already exists
        if 'analysis' in game_data:
            analysis = game_data['analysis']
        else:
            # Get the analysis
            analysis = game_analyzer.analyze_rps_game(game_data)
            
            # Store the analysis in the game data
            game_data['analysis'] = analysis
            storage.update_game(game_id, game_data)
            
        # Format game details
        game_details = {
            'id': game_id,
            'type': game_data['game_type'],
            'start_time': game_data.get('start_time', 'Unknown'),
            'end_time': game_data.get('end_time', 'Unknown'),
            'winner': game_data.get('winner', 'None'),
            'outcome': game_data.get('outcome', 'Unknown')
        }
        
        return render_template('game_analysis.html', 
                             game=game_details,
                             analysis=analysis)
    except Exception as e:
        print(f"Error in view_game_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=str(e)), 500

@analysis_bp.route('/api/analyze-game/<game_id>', methods=['GET'])
def analyze_game(game_id):
    try:
        # Get the game data
        game_data = storage.get_game_as_json(game_id)
        if not game_data:
            return jsonify({'error': 'Game not found'}), 404
        
        # Check if analysis already exists
        if 'analysis' in game_data:
            analysis = game_data['analysis']
        else:
            # Get the analysis
            analysis = game_analyzer.analyze_rps_game(game_data)
            
            # Store the analysis in the game data
            game_data['analysis'] = analysis
            storage.update_game(game_id, game_data)
            
        return jsonify({'analysis': analysis})
    except Exception as e:
        print(f"Error in analyze_game: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/api/analyze-games', methods=['POST'])
def analyze_multiple_games():
    try:
        data = request.json
        game_ids = data.get('game_ids', [])
        question = data.get('question', '')
        game_type = data.get('game_type', 'all')
        
        print(f"Analyzing games: {game_ids}, Game type: {game_type}")
        
        if not game_ids:
            return jsonify({'error': 'No games selected for analysis'}), 400
            
        if not question:
            return jsonify({'error': 'No question provided for analysis'}), 400
        
        # Retrieve game data for the specified IDs
        games_data = []
        for game_id in game_ids:
            game = storage.get_game(game_id)
            if game and (game_type == 'all' or game['game_type'] == game_type):
                games_data.append(game)
        
        if not games_data:
            return jsonify({'error': 'No valid games found for analysis'}), 404
            
        # Determine what type of games we're analyzing (we need consistent types)
        first_game_type = games_data[0]['game_type']
        if not all(game['game_type'] == first_game_type for game in games_data):
            return jsonify({'error': 'All games must be of the same type for analysis'}), 400
            
        # Select the appropriate analysis method based on game type
        if first_game_type == 'rockpaperscissors':
            if len(games_data) == 1:
                # For a single game, use the detailed RPS analysis
                analysis = game_analyzer.analyze_rps_game(games_data[0])
            else:
                # Custom analysis for multiple RPS games
                formatted_games = "\n".join([game_analyzer._format_rps_game(game) for game in games_data])
                prompt = f"""Analyze these Rock Paper Scissors games and answer the following question:
                
                {formatted_games}
                
                User's Question: {question}
                
                Provide a detailed analysis focusing on patterns, strategies, and recommendations.
                """
                
                # Use the new OpenAI API format
                response = game_analyzer.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a Rock Paper Scissors game analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )
                analysis = response.choices[0].message.content.strip()
        else:
            return jsonify({'error': 'Unsupported game type for analysis'}), 400
        
        return jsonify({'analysis': analysis})
    except Exception as e:
        print(f"Error in analyze_multiple_games: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500 