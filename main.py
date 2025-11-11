from model.dataset import load_dictionary, build_graph
from model.gcn_model import GCN
import torch
import torch.nn.functional as F
import torch.optim as optim


def train_gcn(data, num_epochs=200):
    model = GCN(
        input_dim=data.x.size(1),
        hidden_dim=32,
        output_dim=data.x.size(1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Mục tiêu tự giám sát: tái tạo lại đặc trưng ban đầu
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")
    return model


def get_embedding(word, model, data, df):
    """
    Trả về embedding cho từ cụ thể trong cột 'TuNgu'.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

    # Tìm index của từ trong DataFrame
    indices = df.index[df['TuNgu'] == word].tolist()
    if not indices:
        raise ValueError(f"Từ '{word}' không có trong dữ liệu.")
    idx = indices[0]

    return out[idx]


if __name__ == "__main__":
    print("🔹 Đang tải dữ liệu từ điển...")
    df = load_dictionary("data/dic.csv")

    print("🔹 Đang xây dựng đồ thị...")
    data = build_graph(df)

    print("🔹 Đang huấn luyện mô hình GCN...")
    model = train_gcn(data, num_epochs=200)

    word = "ꪀꪱ"
    emb = get_embedding(word, model, data, df)

    print(f"\n✅ Embedding cho '{word}':")
    print(emb)
