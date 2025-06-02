from qdrant_client.http import models
from dotenv import load_dotenv
from .qdrant import get_qdrant_client
from .qdrant import COLLECTION_NAME
import os
import sys

FILE_PATH = "db/myself.txt"

def get_texts_to_embed(file_path): 
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text.split("\n\n")
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def get_embeded_text(file_path):
    texts_to_embed = get_texts_to_embed(FILE_PATH)

    return get_transformer_model().encode(texts_to_embed, batch_size=16, show_progress_bar=True, convert_to_numpy=True)

def save_data(file_path):
    vector_size = get_embeded_text(file_path).shape[1]

    get_qdrant_client().create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )

    points = []
    for idx, (chunk_text, emb) in enumerate(zip(get_texts_to_embed(FILE_PATH), get_embeded_text)):
        points.append(
            models.PointStruct(
                id=idx,                 
                vector=emb.tolist(),    
                payload={
                    "source_file": "junseo.txt",
                    "paragraph_index": idx,     
                    "text": chunk_text
                }
            )
        )

    get_qdrant_client().upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from ai.embedding_model import get_transformer_model

    load_dotenv()

    save_data(FILE_PATH)