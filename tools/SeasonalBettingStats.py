from dotenv import load_dotenv
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
from bs4 import BeautifulSoup
import requests
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import time
import os
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import AgentType


dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
sports_data_key = os.getenv("SPORTS_DATA_IO_API_KEY")

with open('./jsons/team_id.json') as f:
    team_ids = json.load(f)



class SeasonalBettingInput(BaseModel):
    season: str = Field(
        description="""A formatted string of the original question and season, for example: 'question: What tournament team has the best record against the spread this season? : season: 2024""")


class SeasonalBettingStats(BaseTool):
    name = "Seasonal Betting Stats"
    description = """Returns the full list of BetttingEvents for the given season. Intended for those who need to tie BettingEventIDs to GameIDs. Relevant for Futures Feeds and Props Feeds. For example: 'question:  What tournament team has the best record against the spread this season?: season: 2024'"""
    args_schema: Type[BaseModel] = SeasonalBettingInput

    def _run(
            self, param_string: str) -> pd.DataFrame:
        
        # get the abbreviated
        season = param_string.split("season: ")[1].split()[0]
        question = param_string.split("question:")[1]

        print(season)
        print(question)
               
      
        URL = f"https://api.sportsdata.io/v3/cbb/odds/json/BettingEvents/{season}"

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