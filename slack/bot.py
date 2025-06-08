import os
import sys
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai.llm_api import *
from db.retrieve import query

load_dotenv()

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")

app = App(token=SLACK_BOT_TOKEN)

@app.event("app_mention")
def handle_mention(event, say):
    user = event["user"]
    text = event["text"]
    print(f"유저의 질문: {text}")

    try:
        resp = get_open_ai_response(get_prompt(query = text, info = query(text, 5)))
        answer = resp
    except Exception as e:
        answer = f"응답 실패: {e}"

    say(f"<@{user}> {answer}")


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
