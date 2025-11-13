"""
Encoder cho Thai-Vietnamese Seq2Seq model
Sử dụng GCN + LSTM để encode Thai input
"""
import torch
import torch.nn as nn
from .gcn_model import GCN


class Encoder(nn.Module):
    """
    GCN + LSTM Encoder cho Thai text
    - GCN để học graph structure từ dictionary
    - LSTM để encode sequence
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, gcn_hidden_dim=64, 
                 gcn_output_dim=128, num_layers=2, dropout=0.3, graph_data=None):
        """
        Args:
            vocab_size: Kích thước vocabulary của Thai
            embed_size: Kích thước embedding vector
            hidden_size: Kích thước hidden state của LSTM
            gcn_hidden_dim: Kích thước hidden layer của GCN
            gcn_output_dim: Kích thước output của GCN
            num_layers: Số lớp LSTM
            dropout: Tỷ lệ dropout
            graph_data: PyG Data object chứa graph structure (x, edge_index)
        """
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.gcn_output_dim = gcn_output_dim
        
        # GCN model để học graph embeddings
        if graph_data is not None:
            self.use_gcn = True
            gcn_input_dim = graph_data.x.size(1)
            self.gcn = GCN(gcn_input_dim, gcn_hidden_dim, gcn_output_dim)
            self.graph_data = graph_data
            
            # Linear layer để project GCN output về embed_size
            self.gcn_projection = nn.Linear(gcn_output_dim, embed_size)
        else:
            self.use_gcn = False
        
        # Embedding layer (fallback hoặc combine với GCN)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        lstm_input_size = embed_size * 2 if self.use_gcn else embed_size
        
        self.lstm = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def get_gcn_embeddings(self, indices, device):
        """
        Lấy GCN embeddings cho các indices
        
        Args:
            indices: Tensor chứa word indices [batch_size, seq_len]
            device: Device để chuyển tensor
        
        Returns:
            gcn_embeds: [batch_size, seq_len, gcn_output_dim]
        """
        if not self.use_gcn:
            return None
        
        batch_size, seq_len = indices.size()
        
        # Forward pass qua GCN
        x = self.graph_data.x.to(device)
        edge_index = self.graph_data.edge_index.to(device)
        gcn_output = self.gcn(x, edge_index)  # [num_nodes, gcn_output_dim]
        
        # Lấy embeddings cho các indices
        gcn_embeds = []
        for i in range(batch_size):
            batch_embeds = []
            for j in range(seq_len):
                idx = indices[i, j].item()
                if idx < gcn_output.size(0):
                    batch_embeds.append(gcn_output[idx])
                else:
                    # Fallback: zero vector
                    batch_embeds.append(torch.zeros(self.gcn_output_dim, device=device))
            gcn_embeds.append(torch.stack(batch_embeds))
        
        gcn_embeds = torch.stack(gcn_embeds)  # [batch_size, seq_len, gcn_output_dim]
        
        return gcn_embeds
    
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len]
            lengths: Độ dài thực của mỗi sequence [batch_size]
        
        Returns:
            outputs: Output từ tất cả time steps [batch_size, seq_len, hidden_size]
            hidden: Hidden state cuối cùng (h_n, c_n)
                    h_n: [num_layers, batch_size, hidden_size]
                    c_n: [num_layers, batch_size, hidden_size]
        """
        device = x.device
        
        # Standard embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_size]
        
        # Combine with GCN embeddings nếu có
        if self.use_gcn:
            gcn_embeds = self.get_gcn_embeddings(x, device)  # [batch_size, seq_len, gcn_output_dim]
            gcn_embeds = self.gcn_projection(gcn_embeds)  # [batch_size, seq_len, embed_size]
            embedded = torch.cat([embedded, gcn_embeds], dim=2)  # [batch_size, seq_len, embed_size*2]
        
        embedded = self.dropout(embedded)
        
        # LSTM
        if lengths is not None:
            # Pack padded sequence để tối ưu
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            outputs, hidden = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.lstm(embedded)
        
        # outputs: [batch_size, seq_len, hidden_size]
        # hidden: tuple of (h_n, c_n), each [num_layers, batch_size, hidden_size]
        
        return outputs, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Khởi tạo hidden state
        
        Returns:
            (h_0, c_0): Tuple of hidden và cell state
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)
