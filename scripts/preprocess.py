import pandas as pd
import re
import json

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"[\[\]\(\)\-\"\']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()
    return ""

def build_pairs(input_path, output_path="data/pairs.txt"):
    df = pd.read_csv(input_path)
    df["TuNgu"] = df["TuNgu"].apply(clean_text)
    df["NghiaTiengViet"] = df["NghiaTiengViet"].apply(clean_text)

    pairs = []
    for _, row in df.iterrows():
        thai = row["TuNgu"]
        viet_list = re.split(r"[;,/]", row["NghiaTiengViet"])
        for viet in viet_list:
            viet = viet.strip()
            if thai and viet:
                pairs.append(f"{thai}\t{viet}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(pairs)

    print(f"✅ Saved {len(pairs)} pairs to {output_path}")

if __name__ == "__main__":
    build_pairs("data/dic.csv")
