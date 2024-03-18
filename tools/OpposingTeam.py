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

team_ids = {}
with open('./jsons/team_abbr.json') as f:
    team_ids = json.load(f)


class OpposingTeamInput(BaseModel):
    param_string: str = Field(
        description="""A formatted string of the original question, team name, and season, for example: 'question: How did Duke do against the spread this last season? : team_abbr: Duke: opposing_team: North Carolina'""")
    
class OpposingTeamStats(BaseTool): 
    name="OpposingTeamStats"
    description = """Describes recent trends and performance of two teams head-to-head against each other. Useful for tracking head-to-head matchup data and betting data given two specific teams. The input is A formatted string of the original question, team name, and season, for example: 'question: How did Duke do against the spread this last season? : team_abbr: Duke: opposing_team: North Carolina'"""
    args_schema: Type[BaseModel] = OpposingTeamInput

    def _run(
            self, param_string: str) -> pd.DataFrame:
        
        print(param_string)

        # get the abbreviated
        team = param_string.split("team_abbr: ")[
            1].replace("'", "").replace('""', '')
        opposing_team = param_string.split("opposing_team: ")[
            1].replace("'", "").replace('""', '')
        print(team, opposing_team)
        question = param_string.split("question:")[1]
        if team not in team_ids:
            return f"Team {team} not found in the database. Please try again with a different team or check the spelling."
        team_abbr = team_ids[team].strip()
        if opposing_team not in team_ids:
            return f"Team {opposing_team} not found in the database. Please try again with a different team or check the spelling."
        opposing_team_abbr = team_ids[opposing_team].strip()

       

        URL = f"https://api.sportsdata.io/v3/cbb/odds/json/MatchupTrends/{team_abbr}/{opposing_team_abbr}"

        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})
        data_json = data.json()
        data_frames = []
        for data in data_json:
            data_frames.append(pd.DataFrame(data['Teams']))
        df = pd.concat(data_frames)



        df_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo",
                       openai_api_key=open_ai_key),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            openai_api_key=open_ai_key
        )
        question_agent = question + \
            " The dataframe given is a dataframe of head to head matchup data between two teams."
    

        response = df_agent.run(question_agent)
        
        return response