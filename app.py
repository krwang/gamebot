# Main Flask application for RPS AI Game
from flask import Flask, render_template, redirect, url_for, request, jsonify
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Import blueprints
from routes.views import views_bp
from routes.games import games_bp
from routes.analysis import analysis_bp
from routes.game_routes import game_routes
from routes.ai_routes import ai_bp
from routes.stats import stats_bp

# Import middleware
from middleware import track_visitor

# Import Google Sheets manager
from google_sheets import sheets_manager

# Initialize Flask application
app = Flask(__name__)

# Register blueprints
app.register_blueprint(views_bp)
app.register_blueprint(games_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(game_routes)
app.register_blueprint(ai_bp)
app.register_blueprint(stats_bp)

# Add visitor tracking middleware
app = track_visitor(app)

# Create AI models directory if it doesn't exist
os.makedirs('ai_models', exist_ok=True)

@app.route('/')
def index():
    # Redirect to Rock Paper Scissors game by default
    return redirect(url_for('game_routes.play_rps'))

@app.route('/api/record-dennisbot-victory', methods=['POST'])
def record_dennisbot_victory():
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        score = data.get('score')
        ip_address = request.remote_addr  # Get client IP address
        
        if not all([model_name, score]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        success, result = sheets_manager.append_dennisbot_victory(
            model_name=model_name,
            score=score,
            ip_address=ip_address
        )
        
        if success:
            return jsonify({'message': 'Victory recorded successfully'}), 200
        else:
            return jsonify({'error': f'Failed to record victory: {result}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv('FLASK_PORT', 5001))
    app.run(debug=True, port=port) 