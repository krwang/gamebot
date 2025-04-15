import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from game_storage import GameStorage

class RPSDataset(Dataset):
    def __init__(self, game_data):
        self.sequences = []
        self.labels = []
        
        # Convert moves to numerical format
        choice_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
        result_to_idx = {'player_win': 0, 'ai_win': 1, 'tie': 2}
        
        for game in game_data:
            moves = game['moves']
            if len(moves) < 5:  # Skip games with fewer than 5 rounds
                continue
                
            # Create sequences of 5 rounds
            for i in range(len(moves) - 5):
                sequence = []
                for j in range(5):
                    move = moves[i + j]['move_data']
                    player_choice = choice_to_idx[move['player_choice']]
                    ai_choice = choice_to_idx[move['ai_choice']]
                    result = result_to_idx[move['result']]
                    sequence.extend([player_choice, ai_choice, result])
                
                # The label is the player's next move
                next_move = moves[i + 5]['move_data']['player_choice']
                self.sequences.append(sequence)
                self.labels.append(choice_to_idx[next_move])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class RPSPredictor(nn.Module):
    def __init__(self, input_dim=15, embed_dim=64, num_heads=4, num_layers=2, num_classes=3):
        super(RPSPredictor, self).__init__()
        
        # Embedding layer for input sequences
        self.embedding = nn.Embedding(9, embed_dim)  # 9 possible values (3 choices * 3 results)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, sequence_length, embed_dim)
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, embed_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Average pooling over sequence
        x = self.fc(x)
        return x

def train_model(game_data, num_epochs=50, batch_size=32, learning_rate=0.001):
    # Create dataset and dataloader
    dataset = RPSDataset(game_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RPSPredictor().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model

def predict_next_move(model, game_history, device='cpu'):
    """Predict the next move based on the last 5 rounds of game history"""
    choice_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
    result_to_idx = {'player_win': 0, 'ai_win': 1, 'tie': 2}
    idx_to_choice = {0: 'rock', 1: 'paper', 2: 'scissors'}
    
    # Convert last 5 rounds to sequence
    sequence = []
    for move in game_history[-5:]:
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

def load_trained_model(game_id):
    """Load a trained model for a specific game"""
    storage = GameStorage()
    game_data = storage.get_game_as_json(game_id)
    
    if not game_data:
        return None
    
    # Train model on this game's data
    model = train_model([game_data])
    return model 