# embedding.py
from sentence_transformers import SentenceTransformer

def get_transformer_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask", device="cpu")
