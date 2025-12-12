import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel


load_dotenv(override=True)

INSTRUCTIONS = """
You are a software engineer who is expert in designing a roadmap to build an MVP.
You will be provided by the MVP design, technical requirements, and non-technical requirements.
Your task is to design a roadmap step-by-step from the initial stages of building the MVP to the end.
Be consice, don't extract multiple methods to create a step, just the single best suited way.
Try to create a parallel step for various teams to boost the productivity.
Response in Markdown.
"""


class Roadmap(BaseModel):
    roadmap: str = Field(description="A step-by-step roadmap containing what requirements should be met in the process of building the MVP.")
    

openrouter_client = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv('GOOGLE_API_KEY'))
openrouter_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash-preview-09-2025", openai_client=openrouter_client)

roadmap_agent = Agent(
    name="RoadmapAgent",
    instructions=INSTRUCTIONS,
    model=openrouter_model,
    output_type=Roadmap
)