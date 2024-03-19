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

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi



import dotenv
import os
dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASSWORD")

app = Flask(__name__)


@app.route('/chat')
def chat():
    start = time.time()
    question = request.args.get('question')
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
    tools = [LeagueHierarchyTool, TeamTrendsTool,
             PlayerSeasonStatsTool, PlayerGameStatsTool, InternetTool, 
             BettingTrendsByMatchupTool, ScheduleTool, TeamStatisticsTool]

    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate'
    )

    response = agent.invoke({"input": question, 'chat_history': []})

    return jsonify({"response": response, "time": time.time() - start})


if __name__ == '__main__':
    app.run(port=8000, debug=True)
