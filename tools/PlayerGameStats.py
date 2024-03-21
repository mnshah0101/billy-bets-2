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

with open('./jsons/player_ids.json') as f:
    player_ids = json.load(f)

ranked_teams = set([3, 6, 94, 110, 21, 277, 279, 65, 247, 102, 61,
                   268, 26, 108, 335, 267, 106, 334, 275, 220, 336, 111, 96, 212, 256])


def extract_information(text):
    # Regular expressions for each field
    question_pattern = r"question: ([^:]+)"
    player_name_pattern = r"player_name: ([^,]+)"
    season_pattern = r"season: ([^,]+)"
    ranked = r"ranked: ([^,]+)"
    numberofgames_pattern = r"numberofgames: ([^\']+)'"

    # Extracting each field using the regular expressions
    question = re.search(question_pattern, text)
    player_name = re.search(player_name_pattern, text)
    season = re.search(season_pattern, text)
    ranked = re.search(ranked, text)
    numberofgames = re.search(numberofgames_pattern, text)

    # Extracting the matched groups if present
    question_text = question.group(1).strip() if question else None
    player_name_text = player_name.group(1).strip() if player_name else None
    season_text = season.group(1).strip() if season else '2024'
    ranked_text = ranked.group(1).strip() if ranked else 'True'
    numberofgames_text = numberofgames.group(
        1).strip() if numberofgames else 'all'
    if not numberofgames_text:
        numberofgames_text = 'all'

    return question_text, player_name_text, season_text, ranked_text, numberofgames_text


class PlayerGameStatsInput(BaseModel):
    param_string: str = Field(
        description="""A formatted string of the original question, player name, and season, for example: 'question: How many points did Zach Edey average the last 10 games against ranked teams? : player_name: Zach Edey, season: 2023, ranked: True, numberofgames: 3' """)


class PlayerGameStats(BaseTool):
    name = "PlayerGameStats"
    description = """Describes player stats and performance in a single game. Useful for finding player season highs in a specific category, such as points or assists. Also useful for determining a player's performance against a single opponent. The input is a formatted string of the original question, player name, and season, for example: 'question: How many points did Zach Edey average the last 10 games against ranked teams? : player_name: Zach Edey, season: 2023, ranked: True, numberofgames: 3' The current season is 2024.'"""
    args_schema: Type[BaseModel] = PlayerGameStatsInput

    def _run(
            self, param_string: str) -> pd.DataFrame:
        # get the abbreviated
        question, player_name, season, ranked, numberofgames = extract_information(
            param_string)

        if (player_name == None or question == None):
            return "Please provide a question and a player name."

        season = season.strip("'")
        ranked = ranked.strip("'")
        if player_name not in player_ids:
            return f"Player {player_name} not found in the database. Please try again with a different player or check the spelling."
        playerid = player_ids[player_name]

        URL = f"https://api.sportsdata.io/v3/cbb/stats/json/PlayerGameStatsBySeason/{season}/{playerid}/{numberofgames}"

        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})
        df = pd.DataFrame(data.json())
        if ranked == "True":
            df['ranked'] = df['OpponentID'].isin(ranked_teams)
        df_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-4-turbo-preview",
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