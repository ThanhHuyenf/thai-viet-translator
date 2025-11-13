"""Quick test script - Test trên training data"""
import torch
from inference import load_model, translate
from utils.preprocess import load_data

device = torch.device('cpu')
print("Loading model...")
model, thai_vocab, viet_vocab, pos_vocab = load_model('checkpoints/checkpoint_epoch_50.pt', device)

# Load training data
df = load_data('data/dic.csv')

# Test trên toàn bộ training data
print(f"\nTesting on {len(df)} samples")
print("="*80)

correct_word = 0
correct_pos = 0
total = len(df)

for idx in range(total):
    row = df.iloc[idx]
    thai_word = str(row['TuNgu']).strip()
    true_viet = str(row['NghiaTiengViet']).strip()
    true_pos = str(row['LoaiTu']).strip()
    
    pred_viet, pred_pos = translate(model, thai_word, thai_vocab, viet_vocab, pos_vocab, device, max_len=20)
    
    viet_match = "✓" if pred_viet.strip() == true_viet else "✗"
    pos_match = "✓" if pred_pos == true_pos else "✗"
    
    if pred_viet.strip() == true_viet:
        correct_word += 1
    if pred_pos == true_pos:
        correct_pos += 1
    
    # Chỉ in những mẫu sai
    if pred_viet.strip() != true_viet or pred_pos != true_pos:
        print(f"\n{idx+1}. {thai_word}")
        print(f"   Viet: {true_viet:20} | {pred_viet:20} {viet_match}")
        print(f"   POS:  {true_pos:20} | {pred_pos:20} {pos_match}")
    
    if (idx + 1) % 50 == 0:
        print(f"\n[{idx+1}/{total}]...")

print("\n" + "="*80)
print(f"FINAL RESULTS ({total} samples):")
print(f"  Word: {correct_word}/{total} = {100*correct_word/total:.2f}% (Errors: {total - correct_word})")
print(f"  POS:  {correct_pos}/{total} = {100*correct_pos/total:.2f}% (Errors: {total - correct_pos})")
print("="*80)
