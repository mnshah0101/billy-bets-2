import json
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.chat_models import ChatOpenAI
import requests
from typing import Type
import dotenv
import os
from langchain.agents import AgentType
import re


dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
sports_data_key = os.getenv("SPORTS_DATA_IO_API_KEY")

with open('./jsons/team_id.json') as f:
    team_ids = json.load(f)



class TeamStatisticsInput(BaseModel):
    season: str = Field(
        description="""A formatted string of the original question and season, for example: 'question: What is Duke's average 3 point percentage the last 5 games? : season: 2023, team_name : Duke' """)


class TeamStatistics(BaseTool):
    name = "Team Statistics"
    description = """Describes the schedule of games for a given season. Useful for finding team statistics over the past 3, 5, and 10 games, such as team points per game, 3 point percentage, etc. Do not use betting data. The input is a formatted string of the original question and season, for example: 'question: What is Duke's average 3 point percentage the last 5 games? : season: 2023, team_name: Duke Blue Devils' The current season is 2024."""
    args_schema: Type[BaseModel] = TeamStatisticsInput

    def _run(
            self, param_string: str) -> pd.DataFrame:
        
        print(param_string)
        # get the abbreviated
        season = param_string.split("season: ")[1].split()[0]
        team_name = param_string.split("team_name: ")[1].split()[0]
        teamid = team_ids[team_name]
        question = param_string.split("question:")[1]
       
        numberofgames = 'all'
        
      
        URL = f"https://api.sportsdata.io/v3/cbb/scores/json/TeamGameStatsBySeason/{season}/{teamid}/{numberofgames}"

        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})
        print(data)
        data_json = data.json()
        df = pd.DataFrame(data_json)
        df_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-4",
                       openai_api_key=open_ai_key),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            openai_api_key=open_ai_key
        )
        
        response = df_agent.run(question)
    
        return response
