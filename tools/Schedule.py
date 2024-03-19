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

ranked_teams = set([3, 6, 94, 110, 21, 277, 279, 65, 247, 102, 61, 268, 26, 108, 335, 267, 106, 334, 275, 220, 336, 111, 96, 212, 256])


class ScheduleInput(BaseModel):
    season: str = Field(
        description="""A formatted string of the original question and season, for example: 'question: What is Duke's record against ranked ACC teams last season at home? : season: 2023, team_name : Duke Blue Devils, conference: ACC, ranked: True, home_or_away: home' """)


class Schedule(BaseTool):
    name = "Team Wins and Team Conference Wins"
    description = """Describes the schedule of games for a given season. Useful for finding team records against a specific conference (for example, what is a specific team record against the Pac-12), or home and away records. Sum up the ConferenceWins and ConferenceLosses column for the win-loss record and then also make sure to factor in home vs. away. The input is a formatted string of the original question and season, for example: 'question: How did Duke do against ranked ACC teams last season at home? : season: 2023, team_name: Duke Blue Devils, conference: ACC, ranked: True, home_or_away: home' The current season is 2024.'"""
    args_schema: Type[BaseModel] = ScheduleInput

    def _run(
            self, param_string: str) -> pd.DataFrame:
        
        print(param_string)
        # get the abbreviated
        season = param_string.split("season: ")[1].split()[0]
        try:
            ranked = param_string.split("ranked: ")[1].split()[0]
        except (IndexError, AttributeError):  # Catches if the split results in an index error or param_string is None
            ranked = None

        team_name = param_string.split("team_name: ")[1].split()[0]
        teamid = team_ids[team_name]
        if param_string.__contains__('home_or_away:'):
            home_or_away = param_string.split("home_or_away: ")[
                1].replace("'", "").replace('""', '')
        else: 
            home_or_away = None
        numberofgames = 'all'
        pattern = r"conference: ([^,]+)"

        # Search the string using the regular expression
        match = re.search(pattern, param_string)

        # If a match is found, extract the conference name
        if match:
            conference_name = match.group(1).strip()  # Remove any leading/trailing whitespace
        else:
            conference_name = None  # or set to a default value or error message
        question = param_string.split("question:")[1]
      
        URL = f"https://api.sportsdata.io/v3/cbb/scores/json/TeamGameStatsBySeason/{season}/{teamid}/{numberofgames}"

        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})
        print(data)
        data_json = data.json()
        df = pd.DataFrame(data_json)


        if ranked: 
            print("Hi")
            df = df[df['OpponentID'].isin(ranked_teams)]
        #response = df_agent.run(question)
        if home_or_away:
            if home_or_away == 'home':
                print(df)
                df = df[df['HomeOrAway'] == 'HOME']
            elif home_or_away == 'away':
                df = df[df['HomeOrAway'] == 'AWAY']

        #   
        if conference_name: 
            response = f"{team_name} has a record of {df['ConferenceWins'].sum()} - {df['ConferenceLosses'].sum()} against the {conference_name} this season."
        else: 
            response = f"{team_name} has a record of {df['Wins'].sum()} - {df['Losses'].sum()} this season."

        if home_or_away and conference_name: 
            response += f" {team_name} has a record of {df['ConferenceWins'].sum()} - {df['ConferenceLosses'].sum()} at {home_or_away.lower()} this season."
        elif home_or_away: 
            response += f" {team_name} has a record of {df['Wins'].sum()} - {df['Losses'].sum()} at {home_or_away.lower()} this season."
        return response
