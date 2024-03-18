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



class ScheduleInput(BaseModel):
    season: str = Field(
        description="""A formatted string of the original question and season, for example: 'question: What is Duke's record against the ACC last season? : season: 2023, team_name : Duke Blue Devils, conference: ACC' """)


class Schedule(BaseTool):
    name = "Schedule"
    description = """Describes the schedule of games for a given season. Useful for finding team records against a specific conference (for example, what is a specific team record against the Pac-12), or home and away records. You can use conferencewins column and conferenceloss column to output a win-loss record. The input is a formatted string of the original question and season, for example: 'question: How did Duke do against the ACC last season? : season: 2023, team_name: Duke Blue Devils, conference: ACC' The current season is 2024.'"""
    args_schema: Type[BaseModel] = ScheduleInput

    def _run(
            self, param_string: str) -> pd.DataFrame:
        # get the abbreviated
        season = param_string.split("season: ")[1].split()[0]
        team_name = param_string.split("team_name: ")[1].split()[0]
        teamid = team_ids[team_name]
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
        print(season, teamid, numberofgames)
        URL = f"https://api.sportsdata.io/v3/cbb/scores/json/TeamGameStatsBySeason/{season}/{teamid}/{numberofgames}"

        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})
        print(data)
        data_json = data.json()
        df = pd.DataFrame(data_json)
        
        response = f"{team_name} has a record of {df['ConferenceWins'].sum()} - {df['ConferenceLosses'].sum()} against the {conference_name} this season."

        return response
