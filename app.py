from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent
import time
from flask import Flask, request, jsonify
from tools.TeamTrends import TeamTrends
import dotenv
import os
dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")


app = Flask(__name__)


@app.route('/chat')
def chat():
    start = time.time()
    question = request.args.get('question')
    TeamTrendsTool = TeamTrends()
    turbo_llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-4',
        openai_api_key=open_ai_key
    )
    tools = [TeamTrendsTool]
    conversational_agent = initialize_agent(
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=turbo_llm,
        verbose=False)
    response = conversational_agent.run(question)
    return jsonify({"response": response, "time": time.time() - start})


if __name__ == '__main__':
    app.run(port=8000, debug=True)
