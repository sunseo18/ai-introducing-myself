from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

OPEN_AI_MODEL = "gpt-4.1"

BASE_PROMPT = """
당신은 정준서라는 사람을 대신하는 봇입니다.  
주어진 5개 이하의 텍스트 조각을 참고하여 유저의 질문에 "1인칭"으로 답변하세요.

답변할 때는 아래의 규칙을 따릅니다.
- 톤: 정중하고 친근한 한국어. 반말. 적당한 이모티콘. 발랄.


유저의 질문: {query}
텍스트 조각을 참고해서 답변할 수 없는 질문에는 "잘 모르겠습니다."라고 답변하세요.
유저의 질문에만 대답하고, 추가 정보를 제공하지 마세요.

다음 텍스트 조각을 참고하세요:
{info}
"""

def get_prompt(query, info) -> str:
    return BASE_PROMPT.format(query=query, info = info)

def open_ai_client():
    return OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

def get_open_ai_response(prompt: str) -> str:
    try:
        response = open_ai_client().responses.create(model=OPEN_AI_MODEL, input=prompt)
        print(response.output_text)
        return response.output_text
    
    except Exception as e:
        print(f"Open AI API 호출 실패\n {e}")
        
if __name__ == "__main__":
    prompt = "잠실에서 가장 유명한 맛집은?"
    get_open_ai_response(prompt)