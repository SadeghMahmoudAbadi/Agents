import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel
from pydantic import BaseModel, Field


load_dotenv(override=True)

INSTRUCTIONS = """
You are a product designer. Your task is to design a product with the given idea.
You have to expand the idea, add details, and shape it into an MVP.
Be consice, don't recommend multiple designs for the given idea. You must only design the single best MVP.
Response in Markdown.
"""


class Product(BaseModel):
    design: str = Field(description="A detailed design based on the given idea.")
    

openrouter_client = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv('GOOGLE_API_KEY'))
openrouter_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash-preview-09-2025", openai_client=openrouter_client)

product_agent = Agent(
    name="MVPAgent",
    instructions=INSTRUCTIONS,
    model=openrouter_model,
    output_type=Product
)