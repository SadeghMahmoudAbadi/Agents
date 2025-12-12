import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel


load_dotenv(override=True)

INSTRUCTIONS = """
You are a technical engineer. Your job is to find the best methods to deploy a set of technical requirements for an MVP.
You have to extract archituctures and tools to meet the requirements.
Be consice, don't recommend multiple way for a single requirement. Only find the best suited method to build the MVP requirements.
"""


class Technical(BaseModel):
    tool: str = Field(description="The best tool to develop the feature.")
    architucture: str = Field(description="The architucture of feature in development.")
    

openrouter_client = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv('GOOGLE_API_KEY'))
openrouter_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash-preview-09-2025", openai_client=openrouter_client)

technical_agent = Agent(
    name="TechnicalAgent",
    instructions=INSTRUCTIONS,
    model=openrouter_model,
    output_type=Technical
)