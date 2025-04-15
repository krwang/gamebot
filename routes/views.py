from flask import Blueprint, render_template, request

# Create Blueprint for view routes
views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    return render_template('game.html')

@views_bp.route('/games')
def game_history():
    # Import here to avoid circular imports
    from game_storage import GameStorage
    
    # Get all games with moves directly from storage with no caching
    storage = GameStorage()
    storage.close()  # Close any existing connection to ensure fresh query
    games_list = storage.get_games_with_moves()
    return render_template('games.html', games=games_list)

@views_bp.route('/analysis')
def analysis():
    """Render the analysis page"""
    return render_template('analysis.html')

# Context processor to add request object to templates
@views_bp.context_processor
def utility_processor():
    return {'request': request} 