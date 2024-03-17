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


class OpposingTeamInput(BaseModel):
    param_string: str = Field(
        description="""A formatted string of the original question, team name, and season, for example: 'question: How did Duke do against the spread this last season? : team_abbr: Duke'""")
    
class OpposingTeamStats(BaseTool): 
    name="OpposingTeamStats"
    description = """Describes recent trends and performance of two teams head-to-head against each other. Useful for tracking head-to-head matchup and betting data against each other.'"""