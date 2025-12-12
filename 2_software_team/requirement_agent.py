import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel


load_dotenv(override=True)

INSTRUCTIONS = """
You are a requirement engineer. Your job is to extract the requirements for a given MVP.
You have to extract at most 5 technical requirements of the platform.
Be consice, only find the best suited features to build the MVP.
"""


class Feature(BaseModel):
    reason: str = Field(description="The reason this feature is needed to include in the MVP.")
    feature: str = Field(description="The feature needed for the MVP.")


class Requirements(BaseModel):
    features: list[Feature] = Field(description="A list of features required for developing the MVP.")
    

openrouter_client = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv('GOOGLE_API_KEY'))
openrouter_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash-preview-09-2025", openai_client=openrouter_client)

requirement_agent = Agent(
    name="RequirementAgent",
    instructions=INSTRUCTIONS,
    model=openrouter_model,
    output_type=Requirements
)