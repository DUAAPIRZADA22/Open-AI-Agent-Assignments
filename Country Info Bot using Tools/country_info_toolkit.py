from agents import Agent , OpenAIChatCompletionsModel , Runner , function_tool , AsyncOpenAI , RunConfig
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=external_client
)

config=RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True 
)

@function_tool
def get_capital(country : str)->str:
    """Returns the capital of a given country."""
    capitals = {
        "pakistan": "Islamabad",
        "india": "New Delhi",
        "usa": "Washington, D.C.",
        "france": "Paris",
        "japan": "Tokyo"
    }
    return capitals.get(country.lower())

@function_tool
def get_language(country : str)->str:
    """Returns the official language of a given country."""
    languages = {
        "pakistan": "Urdu",
        "india": "Hindi",
        "usa": "English",
        "france": "French",
        "japan": "Japanese"
    }
    return languages.get(country.lower())

@function_tool
def get_population(country : str)->str:
    """Returns the official language of a given country."""
    populations = {
        "pakistan": "241 million",
        "india": "1.4 billion",
        "usa": "331 million",
        "france": "68 million",
        "japan": "125 million"
    }
    return populations.get(country.lower())

orchestrator_agent=Agent(
    name="Get Info",
    instructions="""
        You are a country info bot. When given a country name, use the tools provided to fetch the capital, official language, and population.
        Then, respond with a short paragraph summarizing the country's info.
        """,   
    tools=[get_capital , get_language , get_population]
)

result = Runner.run_sync(
    orchestrator_agent,
    input(" Enter a country name: "),
    run_config=config
)
print('🌍 Country Info:')
print(result.final_output)