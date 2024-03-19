from langchain import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents import AgentType
import time
from flask import Flask, request, jsonify
from tools.TeamTrends import TeamTrends
from tools.LeagueHierarchy import LeagueHierarchy
from tools.Internet import InternetModel
from tools.PlayerSeasonStats import PlayerSeasonStats
from tools.PlayerGameStats import PlayerGameStats
from langchain import hub
from langchain.agents import initialize_agent
from flask_cors import CORS, cross_origin


import dotenv
import os
dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/chat')
@cross_origin()
def chat():
    print('hit')
    start = time.time()
    question = request.args.get('question')
    chat_history = request.args.get('chat_history')
    if (chat_history == None):
        chat_history = []
    print('chat_history')
    print(chat_history)

    TeamTrendsTool = TeamTrends()
    LeagueHierarchyTool = LeagueHierarchy()
    InternetTool = InternetModel()
    PlayerSeasonStatsTool = PlayerSeasonStats()
    PlayerGameStatsTool = PlayerGameStats()
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-4',
        openai_api_key=open_ai_key)
    tools = [LeagueHierarchyTool, TeamTrendsTool,
             PlayerSeasonStatsTool, PlayerGameStatsTool, InternetTool]

    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate'
    )
    chat_history = chat_history[:-5]

    response = agent.invoke(
        {"input": question + " Here is the chat history" + str(chat_history), 'chat_history': chat_history})

    return jsonify({"response": response, "time": time.time() - start})


if __name__ == '__main__':
    app.run(port=8000, debug=True)
