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



class OverUnderInput(BaseModel):
    season: str = Field(
        description="""Provides in-play odds data for a given date. This means odds for games which are in-progress. Only serves the most recently seen data & does not include line movement, for example: 'question: What is Duke's over-under for today's game? : date: 2024-03-24' """)


class OverUnderInStats(BaseTool):
    name = "Over Under Stats"
    description = """Describes the over-under for in-play odds data of games for a given date. Useful for finding over-under data The input is a formatted string of the original question and date, for example: 'question: What is Duke's over-under for today's game? : date: YYYY-MM-DD"""
    args_schema: Type[BaseModel] = OverUnderInput

    def _run(
            self, param_string: str) -> pd.DataFrame:
        
        print(param_string)
        # get the abbreviated
        date = param_string.split("date: ")[1].split()[0]
        question = param_string.split("question:")[1]
        print(date)
        print(question)
      
        URL = f"https://api.sportsdata.io/v3/cbb/odds/json/LiveGameOddsByDate/{date}"

        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})
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
