from ai.embedding_model import get_transformer_model
from dotenv import load_dotenv
from db.qdrant import get_qdrant_client
from db.qdrant import COLLECTION_NAME

def query_junseo_collection(query_text: str, top_k: int = 5):
    
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

    print(f"\n===== Query: {query_text} =====")
    if not results:
        print("조회된 결과가 없습니다.")
        return

    for hit in results:
        payload = hit.payload or {}
        paragraph_text = payload.get("text", "(텍스트 없음)")
        score = hit.score
        print(f"– Score: {score:.4f}\n  Text: {paragraph_text}\n")

if __name__ == "__main__":
    load_dotenv()

    query_junseo_collection("이름이 뭔가요?", top_k=3)