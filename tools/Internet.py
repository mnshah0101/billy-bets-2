from dotenv import load_dotenv
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
from bs4 import BeautifulSoup
import requests
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import time
import os
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

dotenv.load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
sports_data_key = os.getenv("SPORTS_DATA_IO_API_KEY")

load_dotenv()


def get_page_content_requests(url):
    try:
        response = requests.get(url)

        response.raise_for_status()

        # Parse the response text with Beautiful Soup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get all text within the body of the HTML
        body = soup.find('body')

        if body is None:
            return "NOT_ENOUGH_INFORMATION_ERROR"

        return clean_up(body.get_text())

    except requests.exceptions.HTTPError as errh:
        return "An Http Error occurred:" + repr(errh)
    except requests.exceptions.ConnectionError as errc:
        return "A Network Error occurred:" + repr(errc)
    except requests.exceptions.Timeout as errt:
        return "Timeout Error:" + repr(errt)
    except requests.exceptions.RequestException as err:
        return "Something went wrong: " + repr(err)


def clean_up(web_text):
    web_text.replace('\n', '')
    web_text.replace('\t', '')
    # Removes unnecessary whitespaces
    web_text = ' '.join(web_text.split())
    web_text = re.sub(r'\S*@\S*\s?', '', web_text)  # Removes Email
    # Removes Website URL's
    web_text = re.sub(r'http\S+|www.\S+', '', web_text)
    web_text = re.sub(r'@\w+', '', web_text)       # Removes Twitter text
    # Removes button text(ALL CAPS)
    web_text = re.sub(r'\b[A-Z]+\b', '', web_text)

    return web_text


def get_fox_sports_url(link):
    html_doc = requests.get(link).text
    soup = BeautifulSoup(html_doc, 'html.parser')
    scripts = soup.find_all('script', {"type": "application/ld+json"})
    text = ''
    for script in scripts:
        data = json.loads(script.string)
        text += data['articleBody']
    return text


def get_search_string(query):
    try:
        query = query.replace('"', '')
        links = get_links_from_search(query)
        complete_string = ''
        for link in links:
            print('trying link: ', link)
            if ("foxsports" in link):
                complete_string += get_fox_sports_url(link)
            else:
                complete_string += get_page_content_requests(link)
        return complete_string
    except KeyError as err:
        print(err)


def get_answer(question, text):

    turbo_prompt = ChatPromptTemplate.from_template(
        "Based on this question: {question}, process this text from a web page to extract the most important information: {web_page_text}")
    turbo_model = ChatOpenAI(model="gpt-3.5-turbo",
                             api_key=os.getenv('OPENAI_API_KEY'))

    gpt4_prompt_template = ChatPromptTemplate.from_template(
        "Answer this question as an expert sports AI model interacting with a user: {question} using the essential information processed from the web page. If you cannot answer the question, output 'NOT_ENOUGH_INFORMATION_ERROR' word for word. Processed web page text: {processed_text}")
    gpt4_model = ChatOpenAI(model="gpt-4-0125-preview",
                            api_key=os.getenv('OPENAI_API_KEY'))

    output_parser = StrOutputParser()

    try:
        chain_turbo = turbo_prompt | turbo_model | output_parser
        processed_data = chain_turbo.invoke(
            {"question": question, "web_page_text": text})

        chain = gpt4_prompt_template | gpt4_model | output_parser
        output = chain.invoke(
            {"question": question, "processed_text": processed_data})
        return output
    except KeyError as err:
        print(err)
        return "NOT_ENOUGH_INFORMATION_ERROR"


def create_google_query(question):
    prompt = ChatPromptTemplate.from_template(
        "Create the google search query that will find relevant articles to answer this question. Do not provide any explanation. For example, the question: Who are players experts are looking forward? to should yield the search: Kansas college basketball players to look out for. The most current year is 2024. Include the year in your search to get the most pertinent results. Questions should be about college basketball. This is the question: {question}.")
    model = ChatOpenAI(model="gpt-4",
                       api_key=os.getenv('OPENAI_API_KEY'))
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    output = chain.invoke({"question": question})

    return output


def get_links_from_search(query):
    search = GoogleSearchAPIWrapper()
    query = query.replace('"', '')
    response = search.results(query, 10)

    links = [res['link']
             for res in response if 'youtube' and 'wikipedia' and 'kentucky.com/sports' and 'reddit' and 'instagram' and 'video' and 'facebook' and 'twitter' and 'tiktok' not in res['link']]

    return links


def get_subjective_answer(question):
    google_question = create_google_query(question)

    links = get_links_from_search(google_question)

    for link in links:
        print('trying link: ', link)
        text = ''
        if ("foxsports" in link):
            text = get_fox_sports_url(link)
        else:
            text = get_page_content_requests(link)
        answer = get_answer(question, text)
        if ("NOT_ENOUGH_INFORMATION_ERROR" not in answer):
            return answer

    return "Sorry, I don't have enough information to answer that question."


class InternetToolInput(BaseModel):
    query: str = Field(
        description="The Google search query to be used. For example: 'How do experts think Duke will do in their next upcoming game.' ")


class InternetModel(BaseTool):
    name = "Internet Tool"
    description = "Use this tool for expert opinion questions or as a fallback option if none of the other endpoints work. The input is the search query to be used, such as, 'How do experts think Duke will do in their next upcoming game.' The current season is 2024."
    args_schema: Type[BaseModel] = InternetToolInput

    def _run(
            self, query: str) -> pd.DataFrame:

        google_question = create_google_query(query)
        links = get_links_from_search(google_question)
        for link in links:
            print('trying link: ', link)
            text = ''
            if ("foxsports" in link):
                text = get_fox_sports_url(link)
            else:
                text = get_page_content_requests(link)
            answer = get_answer(query, text)
            if ("NOT_ENOUGH_INFORMATION_ERROR" not in answer):
                return answer

        return "Sorry, I don't have enough information to answer that question."
