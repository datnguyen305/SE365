import torch
from torch import nn
import math

class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, config, bias=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(config.hidden_size, config.attention_head_size, bias=bias)
        self.key = nn.Linear(config.hidden_size, config.attention_head_size, bias=bias)
        self.value = nn.Linear(config.hidden_size, config.attention_head_size, bias=bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)