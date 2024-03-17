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


dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
sports_data_key = os.getenv("SPORTS_DATA_IO_API_KEY")


class PlayerSeasonStatsInput(BaseModel):
    param_string: str = Field(
        description="""A formatted string of the original question, player name, and season, for example: 'question: How many points did Zach Edey average last season? : player_name: Zach Edey, season: 2023' """)
    
class PlayerSeasonStats(BaseTool): 
    name="PlayerStats"
    description = """Describes player stats and performance over a given season. Useful for answering questions about seasonal averages of statistical categories, such as points per game, assists, rebounds, etc."""
    args_schema: Type[BaseModel] = PlayerSeasonStatsInput


    def _run(
            self, param_string: str) -> pd.DataFrame:
        # get the abbreviated
        print(param_string)
        season= param_string.split("season: ")[1].split()[0]
        question = param_string.split("question:")[1]
      
        URL = f"https://api.sportsdata.io/v3/nba/stats/json/PlayerSeasonStats/{season}"
        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})
        df = pd.DataFrame(data.json())

        df_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-4",
                       openai_api_key=open_ai_key),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            openai_api_key=open_ai_key
        )
        question_agent = question + " The dataframe given is a dataframe of a players stats within a given season."

        response = df_agent.run(question_agent)

        return response
