"""Inference script để test model đã train"""
import torch
import argparse
from config import Config
from utils.vocabulary import Vocabulary
from utils.preprocess import indices_to_text
from model.encoder import Encoder
from model.decode import GatesSharedDecoder
from model.seq2seq import Seq2Seq
from model.dataset import load_dictionary, build_graph


def load_model(checkpoint_path, device):
    """Load trained model từ checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    vocab_data = checkpoint['vocab_data']
    thai_vocab = vocab_data['thai_vocab']
    viet_vocab = vocab_data['viet_vocab']
    pos_vocab = vocab_data['pos_vocab']
    
    df = load_dictionary(Config.DATA_PATH)
    graph_data = build_graph(df)
    
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    
    return model, thai_vocab, viet_vocab, pos_vocab


def translate(model, thai_word, thai_vocab, viet_vocab, pos_vocab, device, max_len=50):
    """Dịch một từ Thai sang Vietnamese"""
    model.eval()
    
    with torch.no_grad():
        thai_idx = thai_vocab.get_idx(thai_word.strip())
        src = torch.LongTensor([[thai_idx]]).to(device)
        src_len = torch.LongTensor([1]).to(device)
        
        word_predictions, pos_predictions = model.inference(
            src, src_len, max_len=max_len, sos_token=viet_vocab.SOS_token
        )
        
        word_predictions = word_predictions[0]
        pos_predictions = pos_predictions[0]
        
        vietnamese_words = []
        for idx in word_predictions:
            idx = idx.item()
            if idx == viet_vocab.EOS_token:
                break
            if idx not in [viet_vocab.PAD_token, viet_vocab.SOS_token]:
                vietnamese_words.append(viet_vocab.get_word(idx))
        
        pos_tag = None
        for idx in pos_predictions:
            idx = idx.item()
            if idx not in [pos_vocab.PAD_token, pos_vocab.SOS_token, pos_vocab.EOS_token]:
                pos_tag = pos_vocab.get_word(idx)
                break
        
        vietnamese_translation = ' '.join(vietnamese_words)
        
        return vietnamese_translation, pos_tag


def main():
    parser = argparse.ArgumentParser(description='Translate Thai to Vietnamese')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--word', type=str, help='Thai word to translate')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model, thai_vocab, viet_vocab, pos_vocab = load_model(args.checkpoint, device)
    
    if args.interactive:
        # Interactive mode
        print("="*50)
        print("THAI-VIETNAMESE TRANSLATOR (Interactive Mode)")
        print("="*50)
        print("Enter Thai word to translate (or 'quit' to exit)")
        print("-"*50)
        
        while True:
            thai_word = input("\nThai word: ").strip()
            
            if thai_word.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not thai_word:
                continue
            
            translation, pos_tag = translate(
                model, thai_word, thai_vocab, viet_vocab, pos_vocab, device
            )
            
            print(f"Vietnamese: {translation}")
            print(f"POS tag: {pos_tag}")
    
    elif args.word:
        # Single translation
        translation, pos_tag = translate(
            model, args.word, thai_vocab, viet_vocab, pos_vocab, device
        )
        
        print(f"Thai: {args.word}")
        print(f"Vietnamese: {translation}")
        print(f"POS tag: {pos_tag}")
    
    else:
        print("Please specify --word or use --interactive mode")
        print("Example: python inference.py --word 'ꪀꪱ'")
        print("Example: python inference.py --interactive")


if __name__ == '__main__':
    main()
