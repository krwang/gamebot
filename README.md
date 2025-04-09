# Tic Tac Toe with Game Analysis

A Flask-based Tic Tac Toe game with AI-powered game analysis features using OpenAI's API.

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Homebrew** (for macOS):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Python 3** (using Homebrew):
   ```bash
   brew install python
   ```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tictactoe
   ```

2. **Set up Python Virtual Environment**:
   ```bash
   # Create a new virtual environment
   python3 -m venv venv
   
   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   # Make sure your virtual environment is activated
   # The (venv) prefix should appear in your terminal
   python3 -m pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   ```bash
   # Copy the sample environment file
   cp .env.sample .env
   
   # Edit .env and add your OpenAI API key
   # Replace 'your_openai_api_key_here' with your actual API key
   ```

## Running the Application

1. **Activate the virtual environment** (if not already activated):
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # .\venv\Scripts\activate  # On Windows
   ```

2. **Start the Flask application**:
   ```bash
   python3 app.py
   ```

3. **Access the application**:
   Open your web browser and navigate to `http://localhost:5001`

## Features

- Interactive Tic Tac Toe game
- Game history tracking
- AI-powered game analysis using OpenAI
- Ability to resume incomplete games
- Player statistics and insights

## Development

- The application uses Flask for the backend
- Game state is managed in memory with persistent storage
- OpenAI API is used for game analysis features
- Environment variables are managed using python-dotenv

## Troubleshooting

1. **`pip` command not found**:
   - Make sure you've activated the virtual environment
   - Use `python3 -m pip` instead of just `pip`

2. **Module not found errors**:
   - Ensure your virtual environment is activated
   - Try reinstalling dependencies: `python3 -m pip install -r requirements.txt`

3. **Environment variables not loading**:
   - Check that you've copied `.env.sample` to `.env`
   - Verify your OpenAI API key is correctly set in `.env`

## Security Notes

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure
- The `.env` file is included in `.gitignore` 