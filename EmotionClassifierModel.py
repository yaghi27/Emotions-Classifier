
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EmotionClassifierModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, output_dim, dropout, pad_idx, max_length=5000):
        super(EmotionClassifierModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_length + 1)  

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, src):
        batch_size = src.size(0)

        # Get token embeddings
        embedded = self.embedding(src)  
        embedded = self.dropout(embedded)

        # Add [CLS] token
        cls_token = self.cls_token.expand(batch_size, -1, -1) 
        x = torch.cat((cls_token, embedded), dim=1) 

        # Add positional encoding
        x = self.pos_encoder(x) 
        x = x.permute(1, 0, 2) 

        # Transformer encoder
        x = self.transformer_encoder(x) 
        cls_output = x[0]  

        # Normalize and classify
        cls_output = self.layer_norm(cls_output)
        logits = self.fc_out(cls_output)  

        return logits
