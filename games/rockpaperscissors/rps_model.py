import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from game_storage import GameStorage
import tempfile
import os
import logging
import random

# Configure logging\logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RPSDataset(Dataset):
    """
    Creates fixed-length contexts (with padding) of past moves to predict the next player move.
    """
    def __init__(self, game_data, window_size=5):
        self.window_size = window_size
        self.sequences = []
        self.labels = []

        # Mapping for categorical features
        choice_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
        result_to_idx = {'player_win': 0, 'ai_win': 1, 'tie': 2}
        pad_choice = 3
        pad_result = 3

        moves = game_data.get('moves', [])
        move_vectors = []

        for m in moves:
            pm = m['move_data'].get('player_choice') or m['move_data'].get('choice')
            ai = m['move_data'].get('ai_choice')
            res = m['move_data'].get('result')
            if pm and ai and res:
                vec = [
                    choice_to_idx.get(pm, pad_choice),
                    choice_to_idx.get(ai, pad_choice),
                    result_to_idx.get(res, pad_result),
                    m['board_state'].get('player_score', 0),
                    m['board_state'].get('ai_score', 0)
                ]
                move_vectors.append(vec)

        if len(move_vectors) < 2:
            return

        for i in range(len(move_vectors) - 1):
            label = move_vectors[i+1][0]
            start = max(0, i + 1 - self.window_size)
            context = move_vectors[start:i+1]
            if len(context) < self.window_size:
                pad_vec = [pad_choice, pad_choice, pad_result, 0, 0]
                context = [pad_vec] * (self.window_size - len(context)) + context
            self.sequences.append(context)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, lbl

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RPSModel(nn.Module):
    def __init__(self, d_model=16, nhead=1, num_layers=1):
        super().__init__()
        self.player_embed = nn.Embedding(4, d_model, padding_idx=3)
        self.ai_embed = nn.Embedding(4, d_model, padding_idx=3)
        self.result_embed = nn.Embedding(4, d_model, padding_idx=3)
        self.score_linear = nn.Linear(2, d_model)
        self.dropout = nn.Dropout(0.1)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x):
        player = x[:, :, 0]
        ai = x[:, :, 1]
        res = x[:, :, 2]
        scores = x[:, :, 3:].float()

        p_emb = self.player_embed(player)
        ai_emb = self.ai_embed(ai)
        r_emb = self.result_embed(res)
        s_emb = self.score_linear(scores)

        x = p_emb + ai_emb + r_emb + s_emb
        x = self.dropout(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        logits = self.fc(x[:, -1, :])
        return logits

def train_and_save_model(game_data, window_size=5, epochs=30, batch_size=8, lr=5e-4):
    dataset = RPSDataset(game_data, window_size)
    if len(dataset) == 0:
        logger.error("No valid training data found")
        return None

    device = get_device()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RPSModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(dataloader)
            logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.4f}")

    model_path = os.path.join(tempfile.gettempdir(), f"rps_model_{game_data['id']}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    return model_path


def load_model_from_path(model_path):
    model = RPSModel()
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def train_and_load_for_game(game_id, window_size=5):
    storage = GameStorage()
    game_data = storage.get_game_as_json(game_id)
    if not game_data:
        return None, "Game not found"

    path = train_and_save_model(game_data, window_size)
    if not path:
        return None, "Training failed"

    model = load_model_from_path(path)
    return model, None


def predict_next_move(model, game_history, window_size=5):
    choice_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
    result_to_idx = {'player_win': 0, 'ai_win': 1, 'tie': 2}
    idx_to_choice = {0: 'rock', 1: 'paper', 2: 'scissors'}
    pad_choice = 3
    pad_result = 3

    logger.info(f"Predicting next move with history length: {len(game_history)}")
    
    mv = []
    for m in game_history:
        pm = m['move_data'].get('player_choice') or m['move_data'].get('choice')
        ai = m['move_data'].get('ai_choice')
        res = m['move_data'].get('result')
        if pm and ai and res:
            mv.append([
                choice_to_idx.get(pm, pad_choice),
                choice_to_idx.get(ai, pad_choice),
                result_to_idx.get(res, pad_result),
                m['board_state'].get('player_score', 0),
                m['board_state'].get('ai_score', 0)
            ])

    if len(mv) == 0:
        logger.info("No history, returning random choice")
        return random.choice(['rock', 'paper', 'scissors'])

    last = mv[-window_size:]
    if len(last) < window_size:
        pad = [pad_choice, pad_choice, pad_result, 0, 0]
        last = [pad] * (window_size - len(last)) + last

    logger.info(f"Last {window_size} moves: {last}")
    
    tensor = torch.tensor([last], dtype=torch.long)
    with torch.no_grad():
        logits = model(tensor)
        logger.info(f"Raw logits: {logits}")
        
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=-1)
        logger.info(f"Probabilities: {probs}")
        
        # Sample from the probability distribution
        pred = torch.multinomial(probs, 1).item()
        logger.info(f"Sampled prediction: {pred} ({idx_to_choice[pred]})")

    return idx_to_choice[pred]

# Backward-compatible aliases
train_model = train_and_save_model
load_trained_model = load_model_from_path
