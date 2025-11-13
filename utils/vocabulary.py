"""
Vocabulary builder cho Thai-Vietnamese translation
Quản lý từ điển cho Thai words, Vietnamese words, và POS tags
"""
import pickle
from collections import Counter


class Vocabulary:
    """Lớp quản lý từ điển với các token đặc biệt"""
    
    def __init__(self, name="vocab"):
        self.name = name
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.n_words = 0
        
        # Thêm các token đặc biệt
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3
        
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Khởi tạo các token đặc biệt"""
        self.add_word('<PAD>')
        self.add_word('<SOS>')
        self.add_word('<EOS>')
        self.add_word('<UNK>')
    
    def add_word(self, word):
        """Thêm từ vào vocabulary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
        self.word_count[word] += 1
    
    def add_sentence(self, sentence):
        """Thêm câu (danh sách từ) vào vocabulary"""
        for word in sentence.split():
            self.add_word(word)
    
    def get_idx(self, word):
        """Lấy index của từ, trả về UNK nếu không tồn tại"""
        return self.word2idx.get(word, self.UNK_token)
    
    def get_word(self, idx):
        """Lấy từ từ index"""
        return self.idx2word.get(idx, '<UNK>')
    
    def __len__(self):
        return self.n_words
    
    def save(self, filepath):
        """Lưu vocabulary vào file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_count': self.word_count,
                'n_words': self.n_words,
                'name': self.name
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load vocabulary từ file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(name=data['name'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_count = data['word_count']
        vocab.n_words = data['n_words']
        return vocab


def build_vocabularies(df):
    """
    Xây dựng 3 vocabularies từ DataFrame
    
    Args:
        df: DataFrame với các cột TuNgu (Thai), LoaiTu (POS), NghiaTiengViet (Vietnamese)
    
    Returns:
        thai_vocab, viet_vocab, pos_vocab
    """
    thai_vocab = Vocabulary("thai")
    viet_vocab = Vocabulary("vietnamese")
    pos_vocab = Vocabulary("pos")
    
    for _, row in df.iterrows():
        # Thai words
        thai_word = str(row['TuNgu']).strip()
        if thai_word and thai_word != 'nan':
            thai_vocab.add_word(thai_word)
        
        # Vietnamese words
        viet_phrase = str(row['NghiaTiengViet']).strip()
        if viet_phrase and viet_phrase != 'nan':
            # Tách thành các từ riêng lẻ
            for word in viet_phrase.split():
                viet_vocab.add_word(word)
        
        # POS tags
        pos_tag = str(row['LoaiTu']).strip()
        if pos_tag and pos_tag != 'nan':
            pos_vocab.add_word(pos_tag)
    
    print(f"✓ Thai vocabulary size: {len(thai_vocab)}")
    print(f"✓ Vietnamese vocabulary size: {len(viet_vocab)}")
    print(f"✓ POS vocabulary size: {len(pos_vocab)}")
    
    return thai_vocab, viet_vocab, pos_vocab
