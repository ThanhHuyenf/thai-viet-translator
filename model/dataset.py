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


def build_graph(df):
    """
    Build graph từ DataFrame, kết nối các từ có cùng LoaiTu (POS tag)
    """
    num_nodes = len(df)

    edges = []
    # Chỉ sử dụng LoaiTu để build graph (kết nối các từ có cùng POS tag)
    mapping = {}
    for i in range(len(df)):
        val = df.iloc[i]['LoaiTu']
        if pd.isna(val):
            continue
        val = str(val).strip()
        if val not in mapping:
            mapping[val] = []
        mapping[val].append(i)
    
    # Tạo edges giữa các từ có cùng POS tag
    for nodes in mapping.values():
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                edges.append((nodes[a], nodes[b]))
                edges.append((nodes[b], nodes[a]))  # Undirected graph

    if not edges:
        # Nếu không có edge nào, tạo self-loops
        print("⚠️ Không tạo được cạnh từ LoaiTu, tạo self-loops")
        edges = [(i, i) for i in range(num_nodes)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    max_index = int(edge_index.max())
    if max_index >= num_nodes:
        raise ValueError(f"⚠️ Có node index vượt giới hạn: max={max_index}, num_nodes={num_nodes}")

    # Node features: random embeddings
    x = torch.randn((num_nodes, 16))

    data = Data(x=x, edge_index=edge_index)
    return data
