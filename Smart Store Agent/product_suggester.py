import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
import requests

load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")

client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model= OpenAIChatCompletionsModel(
    model="gemini-1.5-flash", 
    openai_client=client,
)

config = RunConfig(
    model=model,
    tracing_disabled=True
)

Product_Suggester_Agent = Agent(
    name="Product Suggester",
    instructions="When a user describes a problem, suggest a product that addresses their need and explain why",
    model=model,
)

input_value = input("Enter your Problem:")

agent_result = Runner.run_sync(Product_Suggester_Agent, input=input_value)