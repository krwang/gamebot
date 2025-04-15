import sqlite3
import json
from datetime import datetime
from pathlib import Path
from threading import local

class GameStorage:
    _thread_local = local()

    def __init__(self, db_path="gamebot.db"):
        """Initialize database path and create tables if they don't exist"""
        self.db_path = Path(db_path)
        self._get_conn()  # Create tables on initialization
        
    def _get_conn(self):
        """Get thread-local database connection"""
        if not hasattr(self._thread_local, 'conn'):
            self._thread_local.conn = sqlite3.connect(str(self.db_path))
            self._thread_local.conn.row_factory = sqlite3.Row
            self.create_tables()
        return self._thread_local.conn

    def create_tables(self):
        """Create the necessary database tables if they don't exist"""
        conn = self._get_conn()
        with conn:
            # Games table stores metadata about each game
            conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    outcome TEXT,
                    winner TEXT,
                    player_symbol TEXT,
                    player_score INTEGER DEFAULT 0,
                    ai_score INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    game_data TEXT
                )
            """)
            
            # Moves table stores each move in every game
            conn.execute("""
                CREATE TABLE IF NOT EXISTS moves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER NOT NULL,
                    player TEXT NOT NULL,
                    position_row INTEGER,
                    position_col INTEGER,
                    move_data TEXT NOT NULL,
                    board_state TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES games (id)
                )
            """)
            
            # Check if all required columns exist, if not add them
            required_columns = {
                'game_type': 'TEXT DEFAULT "rockpaperscissors"',
                'player_score': 'INTEGER DEFAULT 0',
                'ai_score': 'INTEGER DEFAULT 0',
                'draws': 'INTEGER DEFAULT 0',
                'game_data': 'TEXT'
            }
            
            for column, type_def in required_columns.items():
                try:
                    # This query will fail if the column doesn't exist
                    conn.execute(f"SELECT {column} FROM games LIMIT 1")
                except sqlite3.OperationalError:
                    print(f"Adding missing column {column} to games table")
                    conn.execute(f"ALTER TABLE games ADD COLUMN {column} {type_def}")
                    print(f"Added column {column}")
                    conn.commit()

    def start_game(self, game_type, player_symbol: str) -> int:
        """Create a new game record and return its ID"""
        conn = self._get_conn()
        with conn:
            cursor = conn.execute(
                "INSERT INTO games (game_type, start_time, player_symbol) VALUES (?, ?, ?)",
                (game_type, datetime.now().isoformat(), player_symbol)
            )
            return cursor.lastrowid

    def record_move(self, game_id, player, position, board_state, move_data=None):
        """Record a move in the database"""
        row, col = position if position else (None, None)
        move_data_json = json.dumps(move_data) if move_data else json.dumps({})
        
        # Update game scores if they exist in the board state
        if 'player_score' in board_state and 'ai_score' in board_state and 'draws' in board_state:
            conn = self._get_conn()
            with conn:
                conn.execute("""
                    UPDATE games 
                    SET player_score = ?, ai_score = ?, draws = ?
                    WHERE id = ?
                """, (board_state['player_score'], board_state['ai_score'], board_state['draws'], game_id))
        
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT INTO moves 
                (game_id, player, position_row, position_col, move_data, board_state, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id,
                player,
                row,
                col,
                move_data_json,
                json.dumps(board_state),
                datetime.now().isoformat()
            ))

    def end_game(self, game_id, outcome, winner=None):
        """Update game record with final outcome"""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                UPDATE games 
                SET end_time = ?, outcome = ?, winner = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), outcome, winner, game_id))

    def get_game_as_json(self, game_id):
        """Retrieve a complete game record in JSON format"""
        conn = self._get_conn()
        # Get game metadata
        game = conn.execute(
            "SELECT * FROM games WHERE id = ?", (game_id,)
        ).fetchone()
        
        if not game:
            return None
        
        # Get all moves for this game
        moves = conn.execute("""
            SELECT player, position_row, position_col, move_data, board_state, timestamp 
            FROM moves 
            WHERE game_id = ?
            ORDER BY timestamp
        """, (game_id,)).fetchall()
        
        # Get game data if it exists
        game_data = None
        if game["game_data"]:
            try:
                game_data = json.loads(game["game_data"])
            except json.JSONDecodeError:
                game_data = None
        
        # Convert to JSON format
        result = {
            "id": game["id"],
            "game_type": game["game_type"] if "game_type" in game.keys() else "rockpaperscissors",
            "start_time": game["start_time"] if "start_time" in game.keys() else datetime.now().isoformat(),
            "end_time": game["end_time"] if "end_time" in game.keys() else None,
            "outcome": game["outcome"] if "outcome" in game.keys() else None,
            "winner": game["winner"] if "winner" in game.keys() else None,
            "player_symbol": game["player_symbol"] if "player_symbol" in game.keys() else "Player",
            "ai_symbol": "AI",
            "player_score": game["player_score"] if "player_score" in game.keys() else 0,
            "ai_score": game["ai_score"] if "ai_score" in game.keys() else 0,
            "draws": game["draws"] if "draws" in game.keys() else 0,
            "moves": [{
                "player": move["player"],
                "position": [move["position_row"], move["position_col"]] if move["position_row"] is not None else None,
                "move_data": json.loads(move["move_data"]),
                "board_state": json.loads(move["board_state"]),
                "timestamp": move["timestamp"],
                "is_player_move": move["player"] == (game["player_symbol"] if "player_symbol" in game.keys() else "Player")
            } for move in moves]
        }
        
        # Merge game_data if it exists
        if game_data:
            result.update(game_data)
            
        return result

    def get_games_as_json(self, game_type=None, limit=None, offset=0):
        """Retrieve multiple games in JSON format"""
        conn = self._get_conn()
        # Get game IDs with optional filtering by game_type and limit
        query = "SELECT id FROM games"
        params = []
        
        if game_type:
            query += " WHERE game_type = ?"
            params.append(game_type)
            
        query += " ORDER BY start_time DESC"
        
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        game_ids = conn.execute(query, params).fetchall()
        return [self.get_game_as_json(game_id["id"]) for game_id in game_ids]

    def get_games_by_outcome(self, outcome, game_type=None):
        """Retrieve all games with a specific outcome, optionally filtered by game_type"""
        conn = self._get_conn()
        query = "SELECT id FROM games WHERE outcome = ?"
        params = [outcome]
        
        if game_type:
            query += " AND game_type = ?"
            params.append(game_type)
            
        game_ids = conn.execute(query, params).fetchall()
        return [self.get_game_as_json(game_id["id"]) for game_id in game_ids]

    def get_games_with_moves(self, game_type=None):
        """Retrieve all games that have at least one move, optionally filtered by game_type"""
        conn = self._get_conn()
        query = """
            SELECT DISTINCT g.id 
            FROM games g 
            JOIN moves m ON g.id = m.game_id 
        """
        
        params = []
        if game_type:
            query += " WHERE g.game_type = ?"
            params.append(game_type)
            
        query += """
            ORDER BY CASE 
                WHEN g.end_time IS NULL THEN 0 
                ELSE 1 
            END,
            g.start_time DESC
        """
        
        game_ids = conn.execute(query, params).fetchall()
        return [self.get_game_as_json(game_id["id"]) for game_id in game_ids]

    def get_completed_games(self, game_type=None):
        """Retrieve all completed games (games with an end_time), optionally filtered by game_type"""
        conn = self._get_conn()
        query = "SELECT id FROM games WHERE end_time IS NOT NULL"
        params = []
        
        if game_type:
            query += " AND game_type = ?"
            params.append(game_type)
            
        query += " ORDER BY end_time DESC"
        
        game_ids = conn.execute(query, params).fetchall()
        return [self.get_game_as_json(game_id["id"]) for game_id in game_ids]

    def get_active_game(self, game_id):
        """Get a game that hasn't ended yet"""
        conn = self._get_conn()
        game = conn.execute(
            "SELECT * FROM games WHERE id = ? AND end_time IS NULL", 
            (game_id,)
        ).fetchone()
        if game:
            return self.get_game_as_json(game["id"])
        return None
        
    def get_game(self, game_id):
        """Retrieve a game by ID"""
        conn = self._get_conn()
        game = conn.execute(
            "SELECT * FROM games WHERE id = ?", (game_id,)
        ).fetchone()
        
        if not game:
            return None
            
        return dict(game)

    def update_game(self, game_id, game_data):
        """Update a game's data in the database"""
        conn = self._get_conn()
        with conn:
            # Convert game_data to JSON string
            game_data_json = json.dumps(game_data)
            
            # Update the game record
            conn.execute("""
                UPDATE games 
                SET game_data = ?
                WHERE id = ?
            """, (game_data_json, game_id))

    def close(self):
        """Close the database connection for the current thread"""
        if hasattr(self._thread_local, 'conn'):
            self._thread_local.conn.close()
            del self._thread_local.conn 