"""
Seq2Seq model kết nối Encoder (GCN + LSTM) và Decoder (Shared Gates LSTM)
Hỗ trợ Teacher Forcing và multi-task learning (word + POS)
"""
import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    """
    Seq2Seq model cho Thai-Vietnamese translation với POS tagging
    
    Components:
    - Encoder: GCN + LSTM để encode Thai text
    - Decoder: Shared Gates LSTM với attention cho Vietnamese + POS
    """
    
    def __init__(self, encoder, decoder, device):
        """
        Args:
            encoder: Encoder model (GCN + LSTM)
            decoder: Decoder model (Shared Gates LSTM)
            device: cuda hoặc cpu
        """
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, src_len, trg_word, trg_pos, teacher_forcing_ratio=0.5):
        """
        Forward pass với Teacher Forcing
        
        Args:
            src: Thai input [batch_size, src_len]
            src_len: Độ dài thực của Thai sequences [batch_size]
            trg_word: Vietnamese target words [batch_size, trg_len]
            trg_pos: POS tag targets [batch_size, trg_len]
            teacher_forcing_ratio: Tỷ lệ sử dụng teacher forcing (0.0 - 1.0)
        
        Returns:
            word_outputs: [batch_size, trg_len, word_vocab_size]
            pos_outputs: [batch_size, trg_len, pos_vocab_size]
        """
        batch_size = src.size(0)
        trg_len = trg_word.size(1)
        word_vocab_size = self.decoder.fc_word.out_features
        pos_vocab_size = self.decoder.fc_pos.out_features
        
        # Tensors để lưu outputs
        word_outputs = torch.zeros(batch_size, trg_len, word_vocab_size).to(self.device)
        pos_outputs = torch.zeros(batch_size, trg_len, pos_vocab_size).to(self.device)
        
        # 1. ENCODING: Forward qua encoder
        encoder_outputs, encoder_hidden = self.encoder(src, src_len)
        # encoder_outputs: [batch_size, src_len, encoder_hidden_size]
        # encoder_hidden: (h_n, c_n), each [num_layers, batch_size, hidden_size]
        
        # Chuyển encoder_outputs về format [src_len, batch_size, hidden_size] cho decoder
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # 2. Khởi tạo decoder states từ encoder's final hidden state
        # Lấy hidden state từ layer cuối cùng
        h_word = encoder_hidden[0][-1].unsqueeze(0)  # [1, batch_size, hidden_size]
        c_word = encoder_hidden[1][-1].unsqueeze(0)  # [1, batch_size, hidden_size]
        
        # Clone cho POS task
        h_pos = h_word.clone()
        c_pos = c_word.clone()
        
        states_word = (h_word.squeeze(0), c_word.squeeze(0))  # [batch_size, hidden_size]
        states_pos = (h_pos.squeeze(0), c_pos.squeeze(0))
        
        # 3. DECODING: Bắt đầu với <SOS> token
        # Giả sử SOS_token = 1 (được định nghĩa trong vocabulary)
        input_word = trg_word[:, 0].unsqueeze(1)  # [batch_size, 1] - <SOS>
        input_pos = trg_pos[:, 0].unsqueeze(1)    # [batch_size, 1] - <SOS> hoặc first POS
        
        # 4. Decode từng time step
        for t in range(1, trg_len):
            # Forward qua decoder
            word_logits, pos_logits, states_word, states_pos = self.decoder(
                input_word,
                input_pos,
                states_word,
                states_pos,
                encoder_outputs
            )
            
            # Lưu outputs
            word_outputs[:, t, :] = word_logits
            pos_outputs[:, t, :] = pos_logits
            
            # Teacher Forcing: Quyết định sử dụng ground truth hay prediction
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Top prediction
            top_word = word_logits.argmax(1)  # [batch_size]
            top_pos = pos_logits.argmax(1)
            
            # Chọn input cho bước tiếp theo
            if teacher_force:
                input_word = trg_word[:, t].unsqueeze(1)  # [batch_size, 1]
                input_pos = trg_pos[:, t].unsqueeze(1)
            else:
                input_word = top_word.unsqueeze(1)
                input_pos = top_pos.unsqueeze(1)
        
        return word_outputs, pos_outputs
    
    def inference(self, src, src_len, max_len=50, sos_token=1):
        """
        Inference mode (không dùng teacher forcing)
        
        Args:
            src: Thai input [batch_size, src_len]
            src_len: Độ dài thực [batch_size]
            max_len: Độ dài tối đa của output
            sos_token: SOS token index
        
        Returns:
            word_predictions: [batch_size, max_len]
            pos_predictions: [batch_size, max_len]
        """
        batch_size = src.size(0)
        
        # Encoding
        encoder_outputs, encoder_hidden = self.encoder(src, src_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # Khởi tạo decoder states
        h_word = encoder_hidden[0][-1].unsqueeze(0)
        c_word = encoder_hidden[1][-1].unsqueeze(0)
        h_pos = h_word.clone()
        c_pos = c_word.clone()
        
        states_word = (h_word.squeeze(0), c_word.squeeze(0))
        states_pos = (h_pos.squeeze(0), c_pos.squeeze(0))
        
        # Bắt đầu với SOS
        input_word = torch.LongTensor([sos_token] * batch_size).unsqueeze(1).to(self.device)
        input_pos = torch.LongTensor([sos_token] * batch_size).unsqueeze(1).to(self.device)
        
        word_predictions = []
        pos_predictions = []
        
        # Decode
        for t in range(max_len):
            word_logits, pos_logits, states_word, states_pos = self.decoder(
                input_word,
                input_pos,
                states_word,
                states_pos,
                encoder_outputs
            )
            
            # Lấy prediction
            top_word = word_logits.argmax(1)
            top_pos = pos_logits.argmax(1)
            
            word_predictions.append(top_word)
            pos_predictions.append(top_pos)
            
            # Update input cho bước tiếp theo
            input_word = top_word.unsqueeze(1)
            input_pos = top_pos.unsqueeze(1)
        
        # Stack predictions
        word_predictions = torch.stack(word_predictions, dim=1)  # [batch_size, max_len]
        pos_predictions = torch.stack(pos_predictions, dim=1)
        
        return word_predictions, pos_predictions
