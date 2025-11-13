"""
Configuration file cho Thai-Vietnamese translation model
"""


class Config:
    """Hyperparameters và settings cho model"""
    
    # Data paths
    DATA_PATH = 'data/dic.csv'
    SAVE_DIR = 'checkpoints'
    VOCAB_DIR = 'vocab'
    
    # Model hyperparameters
    EMBED_SIZE = 128
    HIDDEN_SIZE = 256
    GCN_HIDDEN_DIM = 64
    GCN_OUTPUT_DIM = 128
    NUM_LAYERS = 1
    DROPOUT = 0.2
    
    # Training hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    TEACHER_FORCING_RATIO = 0.8
    GRADIENT_CLIP = 1.0
    
    # Data split
    TRAIN_SPLIT = 0.9
    VAL_SPLIT = 0.0
    TEST_SPLIT = 0.1
    
    # Sequence parameters
    MAX_LEN = 50
    
    # Loss weights
    WORD_LOSS_WEIGHT = 1.0
    POS_LOSS_WEIGHT = 0.5
    
    # Device
    DEVICE = 'cpu'
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5
    
    # Random seed
    SEED = 42

