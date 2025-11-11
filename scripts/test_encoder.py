import torch
from gensim.models import KeyedVectors
from model.gcn_model import GCNEncoder

wv = KeyedVectors.load_word2vec_format("data/thai_viet_embeddings.vec", binary=False)
embedding_matrix = torch.tensor(wv.vectors)
word2idx = {w: i for i, w in enumerate(wv.key_to_index.keys())}

encoder = GCNEncoder(embedding_matrix)

# ví dụ câu Thái
sentence = ["khau", "nậm", "mương"]
tokens = [word2idx[w] for w in sentence if w in word2idx]
x = torch.tensor(tokens)
edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long)

emb = encoder(x, edge_index)
print("Sentence embedding:", emb.shape)
