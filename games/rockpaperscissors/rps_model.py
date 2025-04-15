import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from game_storage import GameStorage
import tempfile
import os

class RPSDataset(Dataset):
    def __init__(self, game_data, sequence_length=5):
        self.sequences = []
        self.labels = []
        self.sequence_length = sequence_length
        
        # Convert moves to numerical format
        choice_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
        result_to_idx = {'player_win': 0, 'ai_win': 1, 'tie': 2}
        
        for game in game_data:
            moves = game['moves']
            if len(moves) < sequence_length:  # Skip games with fewer rounds than sequence length
                continue
                
            # Create sequences of variable length
            for i in range(len(moves) - sequence_length):
                sequence = []
                for j in range(sequence_length):
                    move = moves[i + j]['move_data']
                    player_choice = choice_to_idx[move['player_choice']]
                    ai_choice = choice_to_idx[move['ai_choice']]
                    result = result_to_idx[move['result']]
                    sequence.extend([player_choice, ai_choice, result])
                
                # The label is the player's next move
                next_move = moves[i + sequence_length]['move_data']['player_choice']
                self.sequences.append(sequence)
                self.labels.append(choice_to_idx[next_move])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class RPSModel(nn.Module):
    def __init__(self):
        super(RPSModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def train_model(game_data):
    try:
        # Convert game moves to training data
        moves = game_data.get('moves', [])
        if not moves:
            return None
            
        # Prepare training data
        X = []
        y = []
        
        for i in range(len(moves) - 1):
            # Convert current move to one-hot encoding
            current_move = moves[i]['player_choice']
            current_encoding = [0, 0, 0]
            current_encoding[current_move] = 1
            X.append(current_encoding)
            
            # Next move as target
            next_move = moves[i + 1]['player_choice']
            next_encoding = [0, 0, 0]
            next_encoding[next_move] = 1
            y.append(next_encoding)
            
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Initialize model
        model = RPSModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
        # Save model
        model_path = os.path.join(tempfile.gettempdir(), f'rps_model_{game_data["id"]}.pth')
        torch.save(model.state_dict(), model_path)
        
        return model_path
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

def load_trained_model(model_path):
    try:
        model = RPSModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_next_move(model, game_history, sequence_length=5, device='cpu'):
    """Predict the next move based on the last N rounds of game history"""
    choice_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
    result_to_idx = {'player_win': 0, 'ai_win': 1, 'tie': 2}
    idx_to_choice = {0: 'rock', 1: 'paper', 2: 'scissors'}
    
    # Convert last N rounds to sequence
    sequence = []
    for move in game_history[-sequence_length:]:
        player_choice = choice_to_idx[move['player_choice']]
        ai_choice = choice_to_idx[move['ai_choice']]
        result = result_to_idx[move['result']]
        sequence.extend([player_choice, ai_choice, result])
    
    # Prepare input tensor
    input_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()
    
    return idx_to_choice[predicted_idx]

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