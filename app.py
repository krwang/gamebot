from flask import Flask
from dotenv import load_dotenv
import os

# Import blueprints
from routes.views import views_bp
from routes.games import games_bp
from routes.analysis import analysis_bp

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Register blueprints
app.register_blueprint(views_bp)
app.register_blueprint(games_bp)
app.register_blueprint(analysis_bp)

if __name__ == "__main__":
    port = int(os.getenv('FLASK_PORT', 5001))
    app.run(debug=True, port=port) 