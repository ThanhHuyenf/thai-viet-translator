from gensim.models import Word2Vec

def train_word2vec(pairs_file="data/pairs.txt", save_path="data/thai_viet_embeddings.vec"):
    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = [line.strip().split("\t") for line in f]

    model = Word2Vec(
        sentences=pairs,
        vector_size=128,
        window=2,
        min_count=1,
        sg=1,
        epochs=100
    )

    model.wv.save_word2vec_format(save_path, binary=False)
    print(f"✅ Saved embeddings to {save_path}")

if __name__ == "__main__":
    train_word2vec()
