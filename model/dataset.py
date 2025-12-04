import pandas as pd
import torch
from torch_geometric.data import Data


def load_dictionary(path='data/dic.csv'):
    """Load từ điển từ CSV file với 3 cột chính: TuNgu, LoaiTu, NghiaTiengViet"""
    df = pd.read_csv(path, on_bad_lines="skip", encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace("'", "").str.replace('"', '')
    
    # Chỉ lấy 3 cột cần thiết và loại bỏ NaN
    df = df[['TuNgu', 'LoaiTu', 'NghiaTiengViet']].dropna()
    
    print(f"✓ Loaded {len(df)} entries for word-by-word translation")
    return df


def build_graph(df, device='cpu'):
    """
    Build graph từ DataFrame, kết nối các từ với POS tag node tương ứng (star topology)
    
    Args:
        df: DataFrame chứa dữ liệu
        device: Device để đẩy graph data lên ('cpu' hoặc 'cuda')
    
    Returns:
        Data object đã được chuyển lên device
    """
    num_word_nodes = len(df)

    edges = []
    # Chỉ sử dụng LoaiTu để build graph (kết nối từ với POS tag node)
    mapping = {}
    pos_tags = []  # Danh sách các POS tag unique
    
    for i in range(len(df)):
        val = df.iloc[i]['LoaiTu']
        if pd.isna(val):
            continue
        val = str(val).strip()
        if val not in mapping:
            mapping[val] = []
            pos_tags.append(val)
        mapping[val].append(i)
    
    # Tổng số nodes = từ + POS tag nodes
    num_pos_nodes = len(pos_tags)
    num_total_nodes = num_word_nodes + num_pos_nodes
    
    # Tạo edges: mỗi từ kết nối với POS tag node tương ứng
    # POS tag nodes có index từ num_word_nodes đến num_word_nodes + num_pos_nodes - 1
    for pos_idx, (pos_tag, word_indices) in enumerate(mapping.items()):
        pos_node_idx = num_word_nodes + pos_idx  # Index của POS tag node
        
        for word_idx in word_indices:
            # Kết nối từ với POS tag node (undirected)
            edges.append((word_idx, pos_node_idx))
            edges.append((pos_node_idx, word_idx))

    if not edges:
        # Nếu không có edge nào, tạo self-loops cho word nodes
        print("⚠️ Không tạo được cạnh từ LoaiTu, tạo self-loops")
        edges = [(i, i) for i in range(num_word_nodes)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    max_index = int(edge_index.max())
    if max_index >= num_total_nodes:
        raise ValueError(f"⚠️ Có node index vượt giới hạn: max={max_index}, num_total_nodes={num_total_nodes}")

    # Node features: random embeddings cho tất cả nodes (word + POS tag nodes)
    x = torch.randn((num_total_nodes, 16))

    data = Data(x=x, edge_index=edge_index)
    
    # Chuyển graph data lên GPU ngay
    data = data.to(device)
    print(f"✓ Graph built: {num_word_nodes} word nodes + {num_pos_nodes} POS tag nodes = {num_total_nodes} total nodes")
    print(f"✓ Graph data moved to {device}")
    
    return data
