"""
Data preprocessing utilities cho Thai-Vietnamese translation
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


def load_data(csv_path='data/dic.csv'):
    """
    Load dữ liệu từ CSV
    
    Returns:
        DataFrame với các cột: TuNgu, LoaiTu, NghiaTiengViet
    """
    # Đọc CSV với quote handling để xử lý dấu phẩy trong giá trị
    df = pd.read_csv(csv_path, on_bad_lines='skip', quotechar='"', encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace("'", "").str.replace('"', '')
    
    # Loại bỏ các hàng có giá trị NaN
    df = df.dropna(subset=['TuNgu', 'LoaiTu', 'NghiaTiengViet'])
    
    # Clean data
    df['TuNgu'] = df['TuNgu'].astype(str).str.strip()
    df['LoaiTu'] = df['LoaiTu'].astype(str).str.strip()
    df['NghiaTiengViet'] = df['NghiaTiengViet'].astype(str).str.strip()
    
    print(f"✓ Loaded {len(df)} entries from {csv_path}")
    return df


def text_to_indices(text, vocab, max_len=None):
    """
    Chuyển đổi text thành list các indices
    
    Args:
        text: Chuỗi text cần chuyển đổi
        vocab: Vocabulary object
        max_len: Độ dài tối đa (padding/truncate)
    
    Returns:
        List of indices
    """
    words = text.split()
    indices = [vocab.get_idx(word) for word in words]
    
    if max_len:
        if len(indices) < max_len:
            # Padding
            indices += [vocab.PAD_token] * (max_len - len(indices))
        else:
            # Truncate
            indices = indices[:max_len]
    
    return indices


def indices_to_text(indices, vocab):
    """
    Chuyển đổi indices thành text
    
    Args:
        indices: List hoặc tensor của indices
        vocab: Vocabulary object
    
    Returns:
        String text
    """
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    
    words = []
    for idx in indices:
        if idx == vocab.EOS_token:
            break
        if idx not in [vocab.PAD_token, vocab.SOS_token]:
            words.append(vocab.get_word(int(idx)))
    
    return ' '.join(words)


class TranslationDataset(Dataset):
    """
    Dataset cho Thai-Vietnamese translation với POS tagging
    """
    
    def __init__(self, df, thai_vocab, viet_vocab, pos_vocab, max_len=50):
        """
        Args:
            df: DataFrame chứa dữ liệu
            thai_vocab: Vocabulary cho tiếng Thái
            viet_vocab: Vocabulary cho tiếng Việt
            pos_vocab: Vocabulary cho POS tags
            max_len: Độ dài tối đa của sequence
        """
        self.df = df.reset_index(drop=True)
        self.thai_vocab = thai_vocab
        self.viet_vocab = viet_vocab
        self.pos_vocab = pos_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            thai_indices: Tensor của Thai word indices
            viet_indices: Tensor của Vietnamese word indices (target)
            pos_indices: Tensor của POS tag indices
            thai_len: Độ dài thực của Thai sequence
            viet_len: Độ dài thực của Vietnamese sequence
        """
        row = self.df.iloc[idx]
        
        # Thai input (single word)
        thai_word = str(row['TuNgu']).strip()
        thai_indices = [self.thai_vocab.get_idx(thai_word)]
        
        # Vietnamese output (có thể là nhiều từ)
        viet_phrase = str(row['NghiaTiengViet']).strip()
        viet_words = viet_phrase.split()
        viet_indices = [self.viet_vocab.SOS_token]
        viet_indices += [self.viet_vocab.get_idx(w) for w in viet_words]
        viet_indices.append(self.viet_vocab.EOS_token)
        
        # POS tag (cũng cần SOS/EOS như Vietnamese)
        pos_tag = str(row['LoaiTu']).strip()
        pos_indices = [self.pos_vocab.SOS_token]
        pos_indices.append(self.pos_vocab.get_idx(pos_tag))
        pos_indices.append(self.pos_vocab.EOS_token)
        
        # Lưu độ dài thực
        thai_len = len(thai_indices)
        viet_len = len(viet_indices)
        
        # Padding
        thai_indices = self._pad_sequence(thai_indices, self.max_len, self.thai_vocab.PAD_token)
        viet_indices = self._pad_sequence(viet_indices, self.max_len, self.viet_vocab.PAD_token)
        pos_indices = self._pad_sequence(pos_indices, self.max_len, self.pos_vocab.PAD_token)
        
        return {
            'thai': torch.LongTensor(thai_indices),
            'viet': torch.LongTensor(viet_indices),
            'pos': torch.LongTensor(pos_indices),
            'thai_len': thai_len,
            'viet_len': viet_len
        }
    
    def _pad_sequence(self, seq, max_len, pad_token):
        """Padding hoặc truncate sequence"""
        if len(seq) < max_len:
            seq += [pad_token] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq


def collate_fn(batch):
    """
    Custom collate function cho DataLoader
    
    Args:
        batch: List of samples từ Dataset
    
    Returns:
        Dictionary of batched tensors
    """
    thai = torch.stack([item['thai'] for item in batch])
    viet = torch.stack([item['viet'] for item in batch])
    pos = torch.stack([item['pos'] for item in batch])
    thai_len = torch.LongTensor([item['thai_len'] for item in batch])
    viet_len = torch.LongTensor([item['viet_len'] for item in batch])
    
    return {
        'thai': thai,
        'viet': viet,
        'pos': pos,
        'thai_len': thai_len,
        'viet_len': viet_len
    }
