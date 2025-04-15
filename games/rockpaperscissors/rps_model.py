import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from game_storage import GameStorage
import tempfile
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RPSDataset(Dataset):
    def __init__(self, game_data):
        self.sequences = []
        self.labels = []
        
        # Convert moves to numerical format
        choice_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
        result_to_idx = {'player_win': 0, 'ai_win': 1, 'tie': 2}
        
        # Get all moves (both player and AI)
        moves = game_data.get('moves', [])
        
        if len(moves) < 2:
            return
            
        # Convert all moves to numerical format
        move_sequence = []
        for move in moves:
            # Get player choice
            player_choice = move['move_data'].get('player_choice') or move['move_data'].get('choice')
            if not player_choice:
                continue
                
            # Get AI choice and result
            ai_choice = move['move_data'].get('ai_choice')
            result = move['move_data'].get('result')
            
            if ai_choice and result:
                # Create a 5-dimensional vector for each move:
                # [player_choice, ai_choice, result, player_score, ai_score]
                move_vector = [
                    choice_to_idx[player_choice],
                    choice_to_idx[ai_choice],
                    result_to_idx[result],
                    move['board_state'].get('player_score', 0),
                    move['board_state'].get('ai_score', 0)
                ]
                move_sequence.append(move_vector)
        
        # Create training samples using the entire sequence
        for i in range(len(move_sequence) - 1):
            # Use all previous moves as context
            context = move_sequence[:i+1]
            next_move = move_sequence[i+1]
            
            # The label is the player's next move (what we want to predict)
            self.sequences.append(context)
            self.labels.append(next_move[0])  # Player's choice is at index 0
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    # Separate sequences and labels
    sequences, labels = zip(*batch)
    
    # Stack sequences and labels
    sequences = torch.stack(sequences)
    labels = torch.stack(labels)
    
    return sequences, labels

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RPSModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super(RPSModel, self).__init__()
        # Input is now 5-dimensional: [player_choice, ai_choice, result, player_score, ai_score]
        self.embedding = nn.Embedding(3, d_model)  # For choices and results
        self.score_embedding = nn.Linear(2, d_model)  # For scores
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            batch_first=True  # Enable batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 3)  # Predict next move
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, 5]
        # Split input into choices/results and scores
        choices = x[:, :, :3]  # [batch_size, seq_len, 3]
        scores = x[:, :, 3:]   # [batch_size, seq_len, 2]
        
        # Embed choices and results
        choices_embedded = self.embedding(choices)  # [batch_size, seq_len, 3, d_model]
        choices_embedded = choices_embedded.mean(dim=2)  # [batch_size, seq_len, d_model]
        
        # Embed scores
        scores_embedded = self.score_embedding(scores.float())  # [batch_size, seq_len, d_model]
        
        # Combine embeddings
        x = choices_embedded + scores_embedded
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Process through transformer (no need to permute since batch_first=True)
        x = self.transformer_encoder(x)
        
        # Use only the last position's output
        x = self.fc(x[:, -1, :])
        return self.softmax(x)

def train_model(game_data):
    try:
        logger.info("Starting model training")
        
        # Create dataset
        dataset = RPSDataset(game_data)
        if len(dataset) == 0:
            logger.error("No valid training data found")
            return None
            
        # Create data loader with batch_size=1 since sequences have different lengths
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Initialize model
        model = RPSModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        logger.info("Starting training loop")
        model.train()
        for epoch in range(100):
            total_loss = 0
            for batch in dataloader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
            
        logger.info("Training completed")
        
        # Save model
        model_path = os.path.join(tempfile.gettempdir(), f'rps_model_{game_data["id"]}.pth')
        logger.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
        
        return model_path
        
    except Exception as e:
        logger.exception("Error during model training")
        return None

def load_trained_model(model_path):
    try:
        model = RPSModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        logger.exception("Error loading model")
        return None

def predict_next_move(model, game_history):
    try:
        # Convert game history to numerical format
        choice_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
        result_to_idx = {'player_win': 0, 'ai_win': 1, 'tie': 2}
        idx_to_choice = {0: 'rock', 1: 'paper', 2: 'scissors'}
        
        # Get all moves
        move_sequence = []
        for move in game_history:
            player_choice = move['move_data'].get('player_choice') or move['move_data'].get('choice')
            ai_choice = move['move_data'].get('ai_choice')
            result = move['move_data'].get('result')
            
            if player_choice and ai_choice and result:
                move_vector = [
                    choice_to_idx[player_choice],
                    choice_to_idx[ai_choice],
                    result_to_idx[result],
                    move['board_state'].get('player_score', 0),
                    move['board_state'].get('ai_score', 0)
                ]
                move_sequence.append(move_vector)
        
        if not move_sequence:
            return 'rock'  # Default to rock if no history
        
        # Convert to tensor
        input_tensor = torch.tensor(move_sequence, dtype=torch.long).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = output.argmax(dim=1).item()
        
        # Return the predicted move directly
        return idx_to_choice[predicted_idx]
        
    except Exception as e:
        logger.exception("Error making prediction")
        return 'rock'  # Default to rock on error

def load_trained_model(game_id, sequence_length=5):
    """Load a trained model for a specific game"""
    storage = GameStorage()
    game_data = storage.get_game_as_json(game_id)
    
    if not game_data:
        return None, "Game not found"
    
    # Train model on this game's data
    model_path = train_model(game_data)
    if not model_path:
        return None, "Error training model"
    
    return model_path, None 