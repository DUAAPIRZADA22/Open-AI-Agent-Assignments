import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing in your .env file.")

# Gemini-compatible OpenAI client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# RunConfig
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Minimal Agent definition
agent = Agent(
    name="Product Suggester Agent",
    instructions="""
You are a helpful assistant that suggests products, remedies, or learning resources based on the user's message.

Examples:
- "I have a headache" → suggest water, oils, or medicine.
- "I want a red dress" → suggest fashion items.
- "I want to learn Python" → suggest courses or books.

Respond as:
Suggestions:
- Suggestion 1 
- Suggestion 2
- Suggestion 3
""",
    model=model
)

# One-time input and output
async def main():
    user_input = input("\nWhats Your Problem: ").strip()

    try:
        result = await Runner.run(agent, user_input, run_config=config)
        print("\n" + result.final_output + "\n")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# Run the script
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
