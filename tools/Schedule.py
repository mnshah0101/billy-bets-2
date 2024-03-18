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
        description='An integer of the season to get the schedule for.')


class Schedule(BaseTool):
    name = "Schedule"
    description = """Describes player stats and performance in a single gasme. Useful for finding player season highs in a specific category, such as points or assists. Also useful for determining a player's performance against a single opponent. The input is a formatted string of the original question, player name, and season, for example: 'question: How many points did Zach Edey average last season? : player_name: Zach Edey, season: 2023' The current season is 2024.'"""
    args_schema: Type[BaseModel] = PlayerGameStatsInput

    def _run(
            self, param_string: str) -> pd.DataFrame:
        # get the abbreviated
        numberofgames = 'all'
        player_name = param_string.split("player_name: ")[
            1].replace("'", "").replace('""', '').strip()
        player_name = player_name.split(',')[0].strip()
        season = param_string.split("season: ")[
            1].replace("'", "").replace('""', '')
        question = param_string.split("question:")[1]

        if player_name not in player_ids:
            return f"Player {player_name} not found in the database. Please try again with a different player or check the spelling."
        playerid = player_ids[player_name]

        URL = f"https://api.sportsdata.io/v3/cbb/stats/json/PlayerGameStatsBySeason/{season}/{playerid}/{numberofgames}"

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
            " The dataframe given is a dataframe of a players stats in every game over a specific season."

        response = df_agent.run(question_agent)

        return response
