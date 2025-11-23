
"""Training script cho Thai-Vietnamese Seq2Seq model"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import random

from config import Config
from utils.vocabulary import build_vocabularies
from utils.preprocess import load_data, TranslationDataset, collate_fn
from model.encoder import Encoder
from model.decode import GatesSharedDecoder
from model.seq2seq import Seq2Seq
from model.dataset import load_dictionary, build_graph


def set_seed(seed):
    """Đặt random seed cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, vocab_data, filepath):
    """Lưu model checkpoint"""
    config_dict = {
        'EMBED_SIZE': Config.EMBED_SIZE,
        'HIDDEN_SIZE': Config.HIDDEN_SIZE,
        'GCN_HIDDEN_DIM': Config.GCN_HIDDEN_DIM,
        'GCN_OUTPUT_DIM': Config.GCN_OUTPUT_DIM,
        'NUM_LAYERS': Config.NUM_LAYERS,
        'DROPOUT': Config.DROPOUT,
        'BATCH_SIZE': Config.BATCH_SIZE,
        'LEARNING_RATE': Config.LEARNING_RATE,
    }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'vocab_data': vocab_data,
        'config': config_dict
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Saved checkpoint: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def calculate_accuracy(outputs, targets, pad_token):
    """Tính accuracy, bỏ qua padding tokens"""
    predictions = outputs.argmax(dim=2)
    mask = (targets != pad_token).float()
    correct = ((predictions == targets).float() * mask).sum()
    total = mask.sum()
    return correct / total if total > 0 else 0


def train_epoch(model, dataloader, optimizer, criterion, device, config, epoch):
    """Train 1 epoch"""
    model.train()
    total_loss = 0
    total_word_acc = 0
    total_pos_acc = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        src = batch['thai'].to(device)
        src_len = batch['thai_len'].to(device)
        trg_word = batch['viet'].to(device)
        trg_pos = batch['pos'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        word_outputs, pos_outputs = model(
            src, src_len, trg_word, trg_pos, 
            teacher_forcing_ratio=config.TEACHER_FORCING_RATIO
        )
        
        word_outputs_flat = word_outputs[:, 1:, :].reshape(-1, word_outputs.size(-1))
        pos_outputs_flat = pos_outputs[:, 1:, :].reshape(-1, pos_outputs.size(-1))
        trg_word_flat = trg_word[:, 1:].reshape(-1)
        trg_pos_flat = trg_pos[:, 1:].reshape(-1)
        
        word_loss = criterion(word_outputs_flat, trg_word_flat)
        pos_loss = criterion(pos_outputs_flat, trg_pos_flat)
        loss = config.WORD_LOSS_WEIGHT * word_loss + config.POS_LOSS_WEIGHT * pos_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Tính accuracy
        batch_size_acc = batch['viet'].size(0)
        seq_len_acc = batch['viet'].size(1) - 1
        
        word_outputs_acc = word_outputs_flat.detach().view(batch_size_acc, seq_len_acc, -1)
        pos_outputs_acc = pos_outputs_flat.detach().view(batch_size_acc, seq_len_acc, -1)
        trg_word_acc = trg_word_flat.view(batch_size_acc, seq_len_acc)
        trg_pos_acc = trg_pos_flat.view(batch_size_acc, seq_len_acc)
        
        word_acc = calculate_accuracy(word_outputs_acc, trg_word_acc, pad_token=0)
        pos_acc = calculate_accuracy(pos_outputs_acc, trg_pos_acc, pad_token=0)
        
        total_word_acc += word_acc
        total_pos_acc += pos_acc
        
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            print(f"  Batch [{batch_idx+1}/{num_batches}] - "
                  f"Loss: {loss.item():.4f} | "
                  f"Word Acc: {word_acc:.4f} | "
                  f"POS Acc: {pos_acc:.4f}")
    
    avg_loss = total_loss / num_batches
    avg_word_acc = total_word_acc / num_batches
    avg_pos_acc = total_pos_acc / num_batches
    
    return avg_loss, avg_word_acc, avg_pos_acc


def evaluate(model, dataloader, criterion, device, config):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_word_acc = 0
    total_pos_acc = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['thai'].to(device)
            src_len = batch['thai_len'].to(device)
            trg_word = batch['viet'].to(device)
            trg_pos = batch['pos'].to(device)
            
            word_outputs, pos_outputs = model(
                src, src_len, trg_word, trg_pos, teacher_forcing_ratio=0
            )
            
            word_outputs_flat = word_outputs[:, 1:, :].reshape(-1, word_outputs.size(-1))
            pos_outputs_flat = pos_outputs[:, 1:, :].reshape(-1, pos_outputs.size(-1))
            trg_word_flat = trg_word[:, 1:].reshape(-1)
            trg_pos_flat = trg_pos[:, 1:].reshape(-1)
            
            word_loss = criterion(word_outputs_flat, trg_word_flat)
            pos_loss = criterion(pos_outputs_flat, trg_pos_flat)
            loss = config.WORD_LOSS_WEIGHT * word_loss + config.POS_LOSS_WEIGHT * pos_loss
            
            total_loss += loss.item()
            
            word_acc = calculate_accuracy(word_outputs[:, 1:, :], trg_word[:, 1:], pad_token=0)
            pos_acc = calculate_accuracy(pos_outputs[:, 1:, :], trg_pos[:, 1:], pad_token=0)
            
            total_word_acc += word_acc
            total_pos_acc += pos_acc
    
    return total_loss / num_batches, total_word_acc / num_batches, total_pos_acc / num_batches


def main():
    """Main training function"""
    set_seed(Config.SEED)
    
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}\n")
    
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.VOCAB_DIR, exist_ok=True)
    
    # Load data and build vocabularies
    print("="*50)
    print("LOADING DATA & BUILDING VOCABULARIES")
    print("="*50)
    df = load_data(Config.DATA_PATH)
    thai_vocab, viet_vocab, pos_vocab = build_vocabularies(df)
    
    thai_vocab.save(os.path.join(Config.VOCAB_DIR, 'thai_vocab.pkl'))
    viet_vocab.save(os.path.join(Config.VOCAB_DIR, 'viet_vocab.pkl'))
    pos_vocab.save(os.path.join(Config.VOCAB_DIR, 'pos_vocab.pkl'))
    
    # Build graph for GCN and move to GPU
    df_for_graph = load_dictionary(Config.DATA_PATH)
    graph_data = build_graph(df_for_graph, device=device)
    print(f"✓ Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Create dataset and split
    print("\n" + "="*50)
    print("CREATING DATASETS")
    print("="*50)
    dataset = TranslationDataset(df, thai_vocab, viet_vocab, pos_vocab, Config.MAX_LEN)
    
    train_size = int(Config.TRAIN_SPLIT * len(dataset))
    val_size = int(Config.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # DataLoader with pin_memory for faster GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, collate_fn=collate_fn,
                             pin_memory=Config.PIN_MEMORY, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, collate_fn=collate_fn,
                           pin_memory=Config.PIN_MEMORY, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_fn,
                            pin_memory=Config.PIN_MEMORY, num_workers=Config.NUM_WORKERS)
    
    # Build model
    print("\n" + "="*50)
    print("BUILDING MODEL")
    print("="*50)
    
    encoder = Encoder(
        vocab_size=len(thai_vocab), embed_size=Config.EMBED_SIZE,
        hidden_size=Config.HIDDEN_SIZE, gcn_hidden_dim=Config.GCN_HIDDEN_DIM,
        gcn_output_dim=Config.GCN_OUTPUT_DIM, num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT, graph_data=graph_data
    )
    
    decoder = GatesSharedDecoder(
        word_vocab_size=len(viet_vocab), pos_vocab_size=len(pos_vocab),
        embed_size=Config.EMBED_SIZE, hidden_size=Config.HIDDEN_SIZE,
        encoder_hidden_size=Config.HIDDEN_SIZE, dropout=Config.DROPOUT
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=viet_vocab.PAD_token)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{Config.NUM_EPOCHS}]")
        print("-" * 50)
        
        train_loss, train_word_acc, train_pos_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, Config, epoch
        )
        
        print(f"\n✓ Train - Loss: {train_loss:.4f} | "
              f"Word Acc: {train_word_acc:.4f} | POS Acc: {train_pos_acc:.4f}")
        
        # Validation
        val_loss, val_word_acc, val_pos_acc = evaluate(
            model, val_loader, criterion, device, Config
        )
        
        print(f"✓ Val   - Loss: {val_loss:.4f} | "
              f"Word Acc: {val_word_acc:.4f} | POS Acc: {val_pos_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                {'thai_vocab': thai_vocab, 'viet_vocab': viet_vocab, 'pos_vocab': pos_vocab},
                os.path.join(Config.SAVE_DIR, 'best_model.pt')
            )
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        if epoch % Config.SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                {'thai_vocab': thai_vocab, 'viet_vocab': viet_vocab, 'pos_vocab': pos_vocab},
                os.path.join(Config.SAVE_DIR, f'checkpoint_epoch_{epoch}.pt')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, Config.NUM_EPOCHS, train_loss,
        {'thai_vocab': thai_vocab, 'viet_vocab': viet_vocab, 'pos_vocab': pos_vocab},
        os.path.join(Config.SAVE_DIR, 'final_model.pt')
    )
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    test_loss, test_word_acc, test_pos_acc = evaluate(
        model, test_loader, criterion, device, Config
    )
    
    print(f"✓ Test - Loss: {test_loss:.4f} | "
          f"Word Acc: {test_word_acc:.4f} | POS Acc: {test_pos_acc:.4f}")
    
    print("\n✓ Training completed!")


if __name__ == '__main__':
    main()
