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


class ScheduleInput(BaseModel):
    season: int = Field(
        description="""A formatted string of the original question and season, for example: 'question: What is Duke and North Carolina's Head to head record over the last 5 games? : season: 2023' """)


class Schedule(BaseTool):
    name = "Schedule"
    description = """Describes the schedule of games for a given season. Useful for finding team records against a specific conference, or home and away records. The input is a formatted string of the original question and season, for example: 'question: How did Duke do against the ACC last season? : season: 2023' The current season is 2024.'"""
    args_schema: Type[BaseModel] = ScheduleInput

    def _run(
            self, param_string: str) -> pd.DataFrame:
        # get the abbreviated
        
        print(param_string)
        season = param_string.split("season: ")[1].split()[0]
        question = param_string.split("question:")[1]
        print(question, season)
        URL = f"https://api.sportsdata.io/v3/cbb/scores/json/SchedulesBasic/{season}"

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
        question_agent = question + \
            " The dataframe given is a dataframe of a team schedule over the entire season."

        response = df_agent.run(question_agent)

        return response
