import re
import pandas as pd
import requests
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
import os
import dotenv
import json
import re
from typing import Type
dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
sports_data_key = os.getenv("SPORTS_DATA_IO_API_KEY")


def extract_fields(text):
    # Regular expressions to extract each field
    team_pattern = r"team: ([^,]+)"
    opposing_team_pattern = r"opposing_team: ([^,]+)"
    question_pattern = r"question: (.+)"

    # Extracting each field using the regular expressions
    team = re.search(team_pattern, text)
    opposing_team = re.search(opposing_team_pattern, text)
    question = re.search(question_pattern, text)

    # Conditional assignment to handle cases where the pattern might not find a match
    team_name = team.group(1).strip() if team else None
    opposing_team_name = opposing_team.group(1).strip() if opposing_team else None
    question_text = question.group(1).strip() if question else None

    return team_name, opposing_team_name, question_text


with open('./jsons/team_abbr.json') as f:
    team_abbr = json.load(f)


class BettingTrendsByMatchupInput(BaseModel):
    param_string: str = Field(description="A param string with both team, opposition team, and original question forrmated like 'team: Duke, opposing_team: North Carolina, question: Based on betting, how do bookmakers have Duke against North Carolina?.'")

class BettingTrendsByMatchup(BaseTool):
    name = "Head to Head Matchups"
    description = "Useful for getting betting trends for any match up between two teams and for current or historical betting trends. Input is a param string with both team, opposition team, and original question forrmated like 'team: Duke, opposing_team: North Carolina, question: Based on betting, how do bookmakers have Duke against North Carolina?''"
    args_schema: Type[BaseModel] = BettingTrendsByMatchupInput
       
    def _run(
            self, param_string: str) -> pd.DataFrame:
       
        team, opponent, question = extract_fields(param_string)
        print(team)
        team = team_abbr[team]
        opp_abbr = team_abbr[opponent]

        URL = f"https://api.sportsdata.io/v3/cbb/odds/json/MatchupTrends/{team}/{opp_abbr}"

        

        data= requests.get(URL, headers={'Ocp-Apim-Subscription-Key': '66b9a43385ed4dc981818ad925d0efb9'})
        data_json = data.json()
        df = pd.DataFrame(data.json()['TeamTrends'])
       
        df_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-4",
                       openai_api_key=open_ai_key),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            openai_api_key=open_ai_key
        )
        question_agent = question + " This includes columns UpcomingGame which give data about upcoming games for both teams and TeamGameTrends which gives data about the last couple of games for the main team but not opposition. Answer the question best as possible."
        response = df_agent.run(question_agent)

        return response