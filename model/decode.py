import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Shared Gates LSTM Cell (Đã sửa đổi) ---
class SharedGatesLSTMCell(nn.Module):
    """
    LSTM cell tùy chỉnh với:
    - Cổng đầu vào (i) và đầu ra (o) được chia sẻ
    - Cổng quên (f) và cell gate (g) riêng cho từng tác vụ
    
    ### THAY ĐỔI: Chấp nhận x_word và x_pos riêng biệt
    """
    def __init__(self, input_size, hidden_size):
        super(SharedGatesLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Cổng i/o chia sẻ 
        self.W_io = nn.Linear(input_size, 2 * hidden_size)
        self.U_io = nn.Linear(hidden_size, 2 * hidden_size)

        # Cổng f/g riêng cho word
        self.W_fg_word = nn.Linear(input_size, 2 * hidden_size)
        self.U_fg_word = nn.Linear(hidden_size, 2 * hidden_size)

        # Cổng f/g riêng cho POS
        self.W_fg_pos = nn.Linear(input_size, 2 * hidden_size)
        self.U_fg_pos = nn.Linear(hidden_size, 2 * hidden_size)

    def forward(self, x_word, x_pos, h_prev_word, c_prev_word, h_prev_pos, c_prev_pos):
        
        io_word_proj = self.W_io(x_word) + self.U_io(h_prev_word)
        io_pos_proj = self.W_io(x_pos) + self.U_io(h_prev_pos)
 
        io_shared = io_word_proj + io_pos_proj 
        
        i, o = torch.chunk(io_shared, 2, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)

        # Tác vụ word (Tính toán riêng biệt) ---
        fg_word = self.W_fg_word(x_word) + self.U_fg_word(h_prev_word)
        f_word, g_word = torch.chunk(fg_word, 2, dim=1)
        f_word = torch.sigmoid(f_word)
        g_word = torch.tanh(g_word)
        c_word = f_word * c_prev_word + i * g_word 
        h_word = o * torch.tanh(c_word)  

        # Tác vụ POS (Tính toán riêng biệt) 
        fg_pos = self.W_fg_pos(x_pos) + self.U_fg_pos(h_prev_pos)
        f_pos, g_pos = torch.chunk(fg_pos, 2, dim=1)
        f_pos = torch.sigmoid(f_pos)
        g_pos = torch.tanh(g_pos)
        c_pos = f_pos * c_prev_pos + i * g_pos 
        h_pos = o * torch.tanh(c_pos)

        return (h_word, c_word), (h_pos, c_pos)


# --- Gates Shared Decoder (Đã sửa đổi) ---
class GatesSharedDecoder(nn.Module):
    """
    Decoder cho multi-task (word + POS) với:
    - Shared Gates LSTM
    - Attention riêng cho từng tác vụ
    - Dropout để giảm overfitting
    """
    def __init__(self, word_vocab_size, pos_vocab_size, embed_size, hidden_size, encoder_hidden_size, dropout=0.3):
        super(GatesSharedDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size

        # Embedding riêng
        self.word_embedding = nn.Embedding(word_vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(pos_vocab_size, embed_size)

        input_size = embed_size + encoder_hidden_size 
        self.lstm_cell = SharedGatesLSTMCell(input_size, hidden_size)

        # Attention riêng cho mỗi tác vụ
        self.attention_word = nn.Linear(hidden_size + encoder_hidden_size, 1)
        self.attention_pos = nn.Linear(hidden_size + encoder_hidden_size, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Linear output riêng
        self.fc_word = nn.Linear(hidden_size, word_vocab_size)
        self.fc_pos = nn.Linear(hidden_size, pos_vocab_size)

    def _calculate_attention(self, hidden_state, encoder_outputs, attention_layer):
        """
        Tính context vector với attention
        - hidden_state: [batch, hidden]
        - encoder_outputs: [seq_len, batch, enc_hidden]
        """
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Lặp lại hidden_state theo seq_len
        repeated_hidden = hidden_state.unsqueeze(0).repeat(seq_len, 1, 1) 
        energy = torch.tanh(attention_layer(torch.cat((repeated_hidden, encoder_outputs), dim=2))) 
        attention_weights = F.softmax(energy.squeeze(2).t(), dim=1) 
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs.permute(1,0,2)).squeeze(1) 
        return context_vector

    def forward(self, prev_word, prev_pos, states_word, states_pos, encoder_outputs):
        """
        Forward pass 1 bước giải mã
        """
        h_prev_word, c_prev_word = states_word
        h_prev_pos, c_prev_pos = states_pos

        # 1. Attention (Riêng biệt)
        context_word = self._calculate_attention(h_prev_word, encoder_outputs, self.attention_word)
        context_pos = self._calculate_attention(h_prev_pos, encoder_outputs, self.attention_pos)

        # 2. Embedding (Riêng biệt)
        word_embedded = self.word_embedding(prev_word).squeeze(1)
        pos_embedded = self.pos_embedding(prev_pos).squeeze(1)

        # 3. Nối embedding + context vector riêng cho từng tác vụ
        lstm_input_word = torch.cat((word_embedded, context_word), dim=1)
        lstm_input_pos = torch.cat((pos_embedded, context_pos), dim=1)
        
        # ### THAY ĐỔI: Áp dụng dropout cho từng luồng riêng biệt
        lstm_input_word = self.dropout(lstm_input_word)
        lstm_input_pos = self.dropout(lstm_input_pos)

        # ### THAY ĐỔI: Không NỐI (concatenate) 2 luồng input
        # lstm_input = torch.cat((lstm_input_word, lstm_input_pos), dim=1)
        
        # 4. Shared Gates LSTM
        # ### THAY ĐỔI: Truyền 2 luồng input riêng biệt vào LSTM cell
        (h_word, c_word), (h_pos, c_pos) = self.lstm_cell(
            lstm_input_word, lstm_input_pos, 
            h_prev_word, c_prev_word, 
            h_prev_pos, c_prev_pos
        )

        h_word = self.dropout(h_word)
        h_pos = self.dropout(h_pos)

        # 5. Output logits
        word_logits = self.fc_word(h_word)
        pos_logits = self.fc_pos(h_pos)

        return word_logits, pos_logits, (h_word, c_word), (h_pos, c_pos)
