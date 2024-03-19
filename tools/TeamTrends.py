import json
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
import requests
from typing import Type
import dotenv
import os
from langchain.agents import AgentType
from dotenv import load_dotenv
import difflib

load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
sports_data_key = os.getenv("SPORTS_DATA_IO_API_KEY")

team_ids = {}
with open('./jsons/team_abbr.json') as f:
    team_ids = json.load(f)


class TeamTrendsInput(BaseModel):
    param_string: str = Field(
        description="A formatted string with the original question and team name, for example: 'question: How did Duke do against the spread this last season? : team_abbr: Duke")


class TeamTrends(BaseTool):
    name = "TeamTrends"
    description = "Describes recent team trends and performance against betting data in recent sets of games for college basketball teams. Useful for answering questions about how teams performed against the spread in the last 3, 5, or 10 games. Treat every question about record to be answered in win-loss format. The input is formatted string of the original question and team name for example: 'question: How did Duke do against the spread this last season? : team_abbr: Duke "
    args_schema: Type[BaseModel] = TeamTrendsInput

    def _run(
            self, param_string: str) -> pd.DataFrame:

        print("TeamTrends tool running")
        # get the abbreviated
        team = param_string.split("team_abbr: ")[
            1].replace("'", "").replace('""', '')
        question = param_string.split("question:")[1]
        if team not in team_ids:
            return f"Team {team} not found in the database. Please try again with a different team or check the spelling."
        team_abbr = team_ids[team]

        URL = f"https://api.sportsdata.io/v3/cbb/odds/json/TeamTrends/{team_abbr}"
        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})
        print(data)
        
        df = pd.DataFrame(data.json()['TeamGameTrends'])
        
        df_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-4",
                       openai_api_key=open_ai_key),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            openai_api_key=open_ai_key
        )
        question_agent = question + " This dataframe has a column called scope which can be either Last 3 Games, Last 3 Away Games, Last 3 Home Games, Last 3 Games as Favorite, and Last 3 Games as Underdog. It also has the same options but for Last 5 Games and Last 10 Games."

        response = df_agent.run(question_agent)

        return response
