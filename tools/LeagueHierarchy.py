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

dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
sports_data_key = os.getenv("SPORTS_DATA_IO_API_KEY")


class LeagueHierarchyInput(BaseModel):
    question: str = Field(description="The question to ask the agent.")


class LeagueHierarchy(BaseTool):
    name = "LeagueHierarchy"
    description = "Useful for getting a college team's current season standings or current season record in the college basketball league standings or for any conference's standings with records. Inputs the user's question as a string."

    def _run(
            self, question: str) -> pd.DataFrame:
        print('league hierarchy called')

        URL = f"https://api.sportsdata.io/v3/cbb/scores/json/LeagueHierarchy"
        data = requests.get(
            URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})

        data_json = data.json()
        data_frames = []
        for data in data_json:
            data_frames.append(pd.DataFrame(data['Teams']))
        df = pd.concat(data_frames)

        df_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-4",
                       openai_api_key=open_ai_key),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            openai_api_key=open_ai_key
        )
        question_agent = question + \
            " The DataFrame given is a DataFrame of college basketball conferences with the league standings of every single conference. Use your prior knowledge of college basketball to ask the agent about the standings of a specific conference or team."
        response = df_agent.run(question_agent)

        return response
