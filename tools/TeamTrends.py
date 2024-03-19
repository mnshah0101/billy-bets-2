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
sports_data_key = "66b9a43385ed4dc981818ad925d0efb9"

team_ids = {}
with open('./jsons/team_abbr.json') as f:
    team_ids = json.load(f)


class TeamTrendsInput(BaseModel):
    param_string: str = Field(
        description="A formatted string with the original question and team name, for example: 'question: How did Duke do against the spread this last season? : team_abbr: Duke")


class TeamTrends(BaseTool):
    name = "TeamTrends"
    description = "Describes recent team trends and performance against betting data in recent sets of games for college basketball teams. Useful for answering questions about how teams performed against the spread and how teams performed in general in the last 3, 5, or 10 games. The input is formatted string of the original question and team name for example: 'question: How did Duke do against the spread this last season? : team_abbr: Duke "
    args_schema: Type[BaseModel] = TeamTrendsInput

    def _run(
            self, param_string: str) -> pd.DataFrame:

        print("TeamTrends tool running")
        # get the abbreviated
        team = param_string.split("team_abbr: ")[
            1].replace("'", "").replace('""', '')
        question = param_string.split("question:")[1]

        listOfTeams= team_ids.keys()
        team = difflib.get_close_matches(team, listOfTeams)[0]
        

        if team not in team_ids:
            return f"Team {team} not found in the database. Please try again with a different team or check the spelling."

        team_abbr = team_ids[team].strip()
        
        

        URL = f"https://api.sportsdata.io/v3/cbb/odds/json/TeamTrends/{team_abbr}"
        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})
        
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
