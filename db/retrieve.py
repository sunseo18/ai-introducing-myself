import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai.embedding_model import get_transformer_model
from db.qdrant import get_qdrant_client
from db.qdrant import COLLECTION_NAME


def query(query_text: str, top_k: int) -> list[str]:
    
    embedded_query = get_transformer_model().encode(
        query_text,
        convert_to_numpy=True
    )
    
    results = get_qdrant_client().search(
        collection_name=COLLECTION_NAME,
        query_vector=embedded_query.tolist(),
        limit=top_k,
        with_payload=True,  
        with_vectors=False  
    )

    print(f"\n===== 질문: {query_text} =====")
    if not results:
        return []

    for hit in results:
        score = hit.score
        text = hit.payload.get("text", "정보 없음")
        print(f"– 유사도: {score:.4f}\n  결과: {text}\n")

    return [hit.payload["text"] for hit in results]

if __name__ == "__main__":

    load_dotenv()

    query("이름이 뭔가요?", top_k=3)
