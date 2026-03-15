import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# --- MODULE: VARIATIONAL DROPOUT (Giữ nguyên) ---
class VariationalDropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        # x shape: (Batch, Dim) -> Dùng cho Hidden State
        mask = x.new_empty(x.size(0), x.size(1)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)
        return x * mask

# ==========================================
# 1. LSTM MODEL (Task 1)
# Áp dụng Variational Dropout cho hidden state cuối cùng
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # LSTM nhận trực tiếp input_dim
        # Lưu ý: Nếu layer_dim > 1, cần xử lý dropout giữa các layer tùy theo yêu cầu (nn.LSTM hỗ trợ dropout giữa các tầng, nhưng không phải VD).
        # Ta chỉ thay thế nn.Dropout ở đầu ra bằng VariationalDropout.
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Thay thế Standard Dropout bằng Variational Dropout
        self.dropout = VariationalDropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Lấy hidden state cuối cùng
        out = out[:, -1, :]
        
        # Áp dụng Variational Dropout
        out = self.dropout(out)
        out = self.fc(out) 
        return out

# ==========================================
# 2. SEQ2SEQ MODEL (Task 2)
# Áp dụng Variational Dropout cho hidden state của Decoder (recurrently)
# ==========================================
class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_seq_len, output_dim, dropout=0.2):
        super(Seq2SeqModel, self).__init__()
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout_p = dropout
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.decoder_cell = nn.LSTMCell(output_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Khai báo Variational Dropout module (dùng để tham chiếu p)
        self.variational_dropout = VariationalDropout(dropout)
        
    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        # 1. ENCODER
        _, (hidden, cell) = self.encoder(x)
        
        if self.layer_dim > 1:
            h_t, c_t = hidden[-1], cell[-1]
        else:
            h_t, c_t = hidden.squeeze(0), cell.squeeze(0)
            
        # Tạo Variational Mask cho Decoder (recurrent dropout: mask được tạo 1 lần và tái sử dụng)
        drop_mask = None
        if self.training and self.dropout_p > 0:
            p = self.dropout_p
            # Tạo mask với kích thước (Batch, Hidden_Dim)
            drop_mask = h_t.new_empty(h_t.size()).bernoulli_(1 - p) / (1 - p)

        # 2. DECODER LOOP
        decoder_input = x[:, -1, -1].unsqueeze(1) 
        outputs = []
        
        for t in range(self.output_seq_len):
            h_t, c_t = self.decoder_cell(decoder_input, (h_t, c_t))
            
            # Apply Variational Dropout (reusing the static mask)
            if drop_mask is not None:
                h_t_drop = h_t * drop_mask
            else:
                h_t_drop = h_t
                
            prediction = self.fc(h_t_drop)
            outputs.append(prediction)
            
            if target is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1) 
            else:
                decoder_input = prediction
            
        return torch.cat(outputs, dim=1)

# ==========================================
# 3. LUONG ATTENTION MODEL (Task 2)
# ==========================================
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        # General Score: h_t * W * h_s
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (Batch, Hidden) -> h_t hiện tại
        # encoder_outputs: (Batch, Seq, Hidden)
        
        # Tính Score
        # query: (Batch, Hidden, 1)
        query = self.W(decoder_hidden).unsqueeze(2)
        
        # scores: (Batch, Seq, 1)
        scores = torch.bmm(encoder_outputs, query).squeeze(2)
        
        # Weights: (Batch, Seq)
        weights = F.softmax(scores, dim=1)
        
        # Context Vector: (Batch, 1, Hidden) -> squeeze -> (Batch, Hidden)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, weights

class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_seq_len, output_dim, dropout=0.2):
        super(Seq2SeqAttentionModel, self).__init__()
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout_p = dropout

        # Encoder (Direct Input)
        self.encoder = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Luong Attention
        self.attention = LuongAttention(hidden_dim)
        
        # Decoder Cell
        self.decoder_cell = nn.LSTMCell(output_dim, hidden_dim)
        
        # Lớp trộn (Concat) giữa Hidden state và Context
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Khai báo Variational Dropout module
        self.variational_dropout = VariationalDropout(dropout)
        
    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        # 1. ENCODER
        encoder_outputs, (hidden, cell) = self.encoder(x)
        
        if self.layer_dim > 1:
            h_t, c_t = hidden[-1], cell[-1]
        else:
            h_t, c_t = hidden.squeeze(0), cell.squeeze(0)
            
        # Tạo Variational Mask (tái sử dụng mask)
        drop_mask = None
        if self.training and self.dropout_p > 0:
            p = self.dropout_p
            drop_mask = h_t.new_empty(h_t.size()).bernoulli_(1 - p) / (1 - p)
            
        # 2. DECODER LOOP
        decoder_input = x[:, -1, -1].unsqueeze(1) 
        outputs = []
        
        for t in range(self.output_seq_len):
            # A. Chạy LSTM trước để lấy trạng thái h_t hiện tại
            h_t, c_t = self.decoder_cell(decoder_input, (h_t, c_t))
            
            # B. Tính Attention dựa trên h_t MỚI
            context, _ = self.attention(h_t, encoder_outputs)
            
            # C. Kết hợp thông tin: [h_t; context] -> Tanh -> vector hỗn hợp
            concat_input = torch.cat((h_t, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            
            # D. Apply Recurrent Variational Dropout lên vector hỗn hợp
            if drop_mask is not None:
                concat_output = concat_output * drop_mask
                
            # E. Dự báo
            prediction = self.fc(concat_output)
            outputs.append(prediction)
            
            if target is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1) 
            else:
                decoder_input = prediction
            
        return torch.cat(outputs, dim=1)