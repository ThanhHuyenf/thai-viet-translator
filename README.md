# Thai-Vietnamese Translator

Hệ thống dịch từ Thái Lan sang Tiếng Việt sử dụng mô hình Seq2Seq với GCN và Shared Gates LSTM.

## Tính năng

- **Multi-task Learning**: Dịch từ + Gán nhãn POS cùng lúc
- **GCN Encoder**: Sử dụng Graph Convolutional Network để học biểu diễn từ vựng
- **Shared Gates LSTM**: LSTM cell tùy chỉnh với cổng chia sẻ giữa các tác vụ
- **Attention Mechanism**: Attention riêng cho từng tác vụ (word, POS)

## Kiến trúc

```
Encoder (GCN + Bi-LSTM) → Shared Gates Decoder → [Word Output, POS Output]
```

### Thành phần chính:

- **Encoder**: GCN + Bidirectional LSTM với 2 layers
- **Decoder**: Shared Gates LSTM với attention riêng cho word và POS
- **Multi-task**: Dự đoán cả từ tiếng Việt và nhãn POS

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
thai-viet-translator/
├── config.py              # Cấu hình hyperparameters
├── train.py               # Script huấn luyện
├── inference.py           # Script dịch/inference
├── quick_test.py          # Test nhanh
├── requirements.txt       # Dependencies
├── data/
│   ├── dic.csv           # Thai-Vietnamese dictionary
│   └── sentence_pairs.csv
├── model/
│   ├── encoder.py        # GCN + LSTM Encoder
│   ├── decode.py         # Shared Gates Decoder
│   ├── seq2seq.py        # Seq2Seq model
│   ├── gcn_model.py      # GCN layers
│   └── dataset.py        # Dataset và graph builder
└── utils/
    ├── preprocess.py     # Xử lý dữ liệu
    └── vocabulary.py     # Vocabulary class
```

## Sử dụng

### Huấn luyện

```bash
python train.py
```

Model sẽ được lưu tại `checkpoints/`:
- `best_model.pt`: Model tốt nhất (validation loss thấp nhất)
- `final_model.pt`: Model cuối cùng
- `checkpoint_epoch_*.pt`: Checkpoints theo epoch

### Dịch từ

**Dịch một từ:**
```bash
python inference.py --word "ꪀꪱ"
```

**Chế độ tương tác:**
```bash
python inference.py --interactive
```

## Hyperparameters

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| EMBED_SIZE | 128 | Kích thước embedding |
| HIDDEN_SIZE | 256 | Kích thước hidden state |
| GCN_HIDDEN_DIM | 64 | Hidden dimension của GCN |
| BATCH_SIZE | 16 | Batch size |
| LEARNING_RATE | 0.001 | Learning rate |
| NUM_EPOCHS | 30 | Số epochs |
| DROPOUT | 0.3 | Dropout rate |

Xem chi tiết trong `config.py`.

## Kết quả

Model dự đoán:
- **Vietnamese Word**: Từ tiếng Việt tương ứng
- **POS Tag**: Nhãn từ loại (Noun, Verb, Adj, etc.)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- CUDA (khuyến nghị cho GPU training)

## License

MIT
