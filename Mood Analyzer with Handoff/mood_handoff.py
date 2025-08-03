

import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import asyncio

# Load .env and get API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing in your .env file.")

# Gemini-compatible client setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Happy Agent
happy_agent = Agent(
    name="Happy Agent",
    instructions="""
You are the Happy Agent.
Respond with one cheerful, motivating, or fun activity suggestion that matches a happy or joyful mood.
Start your response with: "Happy Agent tool called"
""",
    model=model
)

# Sad Agent
sad_agent = Agent(
    name="Sad Agent",
    instructions="""
You are the Sad Agent.
Respond with one comforting or uplifting activity to help someone feel better when they are sad or down.
Start your response with: "Sad Agent tool called"
""",
    model=model
)

# Mood Analyzer Agent (Triage)
triage_agent = Agent(
    name="Mood Analyzer",
    instructions="""
You are a Mood Analyzer.
Analyze the user's message and decide their mood.

If the user is happy, excited, or joyful, handoff to the 'Happy Agent'.
If the user is sad, low, stressed, or heartbroken, handoff to the 'Sad Agent'.

Do NOT respond yourself. Only forward the message to the correct agent.
""",
    model=model,
    handoffs=[happy_agent, sad_agent]
)

# Main runner
async def main():
    user_input = input("How are you feeling today? ").strip()
    try:
        result = await Runner.run(triage_agent, user_input, run_config=config)
        print("\n" + result.final_output + "\n")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# Run
if __name__ == "__main__":
    asyncio.run(main())
