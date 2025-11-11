import pandas as pd
import torch
from torch_geometric.data import Data


def load_dictionary(path='data/dic.csv'):
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.replace("'", "").str.replace('"', '')
    return df


def build_graph(df):
    num_nodes = len(df)

    edges = []
    for col in ['LoaiTu', 'PhaiSinh']:
        mapping = {}
        for i in range(len(df)):
            val = df.iloc[i][col]
            if pd.isna(val):
                continue
            if val not in mapping:
                mapping[val] = []
            mapping[val].append(i)
        for nodes in mapping.values():
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    edges.append((nodes[a], nodes[b]))

    if not edges:
        raise ValueError("⚠️ Không tạo được cạnh nào — có thể dữ liệu trống hoặc thiếu cột.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    max_index = int(edge_index.max())
    if max_index >= num_nodes:
        raise ValueError(f"⚠️ Có node index vượt giới hạn: max={max_index}, num_nodes={num_nodes}")

    x = torch.randn((num_nodes, 16))

    data = Data(x=x, edge_index=edge_index)
    return data
