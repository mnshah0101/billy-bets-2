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
from tools.BettingTrendsByMatchup import BettingTrendsByMatchup
from tools.Schedule import Schedule
from tools.TeamStatistics import TeamStatistics
from langchain import hub
from langchain.agents import initialize_agent
from flask_cors import CORS, cross_origin

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


import dotenv
import os
dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASSWORD")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@cross_origin()
def index():
    return "Hi I'm Billy"


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
    BettingTrendsByMatchupTool = BettingTrendsByMatchup()
    ScheduleTool = Schedule()
    TeamStatisticsTool = TeamStatistics()
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-4',
        openai_api_key=open_ai_key)
    tools = [ScheduleTool, LeagueHierarchyTool, TeamTrendsTool,
             PlayerGameStatsTool,
             BettingTrendsByMatchupTool, PlayerSeasonStatsTool, TeamStatisticsTool, InternetTool,]

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
        {"input": "Answer this question about mens college basketball. The current season is 2024. " + question + " Here is the chat history: " + str(chat_history), 'chat_history': []})

    client = MongoClient(os.getenv("MONGO_URI"), server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    db = client['billybets']
    collection = db['results']

    try:
        doc = {'question': question,
               'answer': response['output'], 'time': time.time() - start}
        collection.insert_one(doc)
    except Exception as e:
        print(e)

    return jsonify({"response": response, "time": time.time() - start})


if __name__ == '__main__':
    app.run(port=8000, debug=True)
