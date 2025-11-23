"""
Configuration file cho Thai-Vietnamese translation model
"""


class Config:
    """Hyperparameters và settings cho model"""
    
    # Data paths
    DATA_PATH = 'data/dic.csv'
    SAVE_DIR = 'checkpoints'
    VOCAB_DIR = 'vocab'
    
    # Model hyperparameters (điều chỉnh cho 13,000 từ)
    EMBED_SIZE = 128         # Tăng lên cho vocab lớn
    HIDDEN_SIZE = 256        # Tăng capacity
    GCN_HIDDEN_DIM = 64      # Tăng GCN capacity
    GCN_OUTPUT_DIM = 128     # Tăng output
    NUM_LAYERS = 2           # Thêm layer cho model phức tạp hơn
    DROPOUT = 0.3            # Giữ dropout cao với data nhiều
    
    # Training hyperparameters (cho 13K từ)
    BATCH_SIZE = 64          # Tăng batch size với data lớn
    LEARNING_RATE = 0.001    # Learning rate chuẩn
    NUM_EPOCHS = 30          # Giảm epochs, với 13K data train nhanh hơn
    TEACHER_FORCING_RATIO = 0.7  # Tăng teacher forcing
    GRADIENT_CLIP = 1.0
    
    # Data split (với 13K samples: ~10.4K train, ~1.3K val, ~1.3K test)
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Sequence parameters (cho word-by-word translation)
    MAX_LEN = 20             # Giảm từ 50, vì dịch từng từ nên output ngắn
    
    # Loss weights
    WORD_LOSS_WEIGHT = 1.0
    POS_LOSS_WEIGHT = 0.5
    
    # Device
    DEVICE = 'cpu'
    
    # Logging (với ~183 batches/epoch ở batch_size=64)
    LOG_INTERVAL = 20        # Log mỗi 20 batches
    SAVE_INTERVAL = 5        # Save mỗi 5 epochs
    
    # Random seed
    SEED = 42

