from flask import Flask, render_template, redirect, url_for
from dotenv import load_dotenv
import os

# Import blueprints
from routes.views import views_bp
from routes.games import games_bp
from routes.analysis import analysis_bp
from routes.game_routes import game_routes
from routes.ai_routes import ai_bp

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Register blueprints
app.register_blueprint(views_bp)
app.register_blueprint(games_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(game_routes)
app.register_blueprint(ai_bp)

# Create AI models directory if it doesn't exist
os.makedirs('ai_models', exist_ok=True)

@app.route('/')
def index():
    # Redirect to Rock Paper Scissors game by default
    return redirect(url_for('game_routes.play_rps'))

if __name__ == "__main__":
    port = int(os.getenv('FLASK_PORT', 5001))
    app.run(debug=True, port=port) 