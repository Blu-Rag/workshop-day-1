from dotenv import load_dotenv
import requests
from rich import print
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
import os

load_dotenv()

llm = init_chat_model(
    model="mistral-medium-latest", model_provider="mistralai", max_tokens=512
)


@tool
def get_weather(city: str) -> str:
    """
    Use this tool to get latest data about the weather of various cities all over the world.
    """

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {"q": city, "units": "metric", "appid": os.getenv("OPENWEATHER_API_KEY")}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return "Could not fetch weather."

    data = response.json()

    return (
        f"Temperature: {data['main']['temp']}°C\n"
        f"Humidity: {data['main']['humidity']}%\n"
        f"Condition: {data['weather'][0]['description']}\n"
        f"Wind Speed: {data['wind']['speed']} m/s"
    )


tavily_search = TavilySearch(
    max_results=3,
    search_depth="basic",
)

agent = create_agent(
    model="mistral-medium-latest",
    tools=[get_weather, tavily_search],
    system_prompt="You are a weather and news report journalist with over 10+ yrs of experience.",
)

while True:
    user_input = input("User: ")
    print("\n" + "=" * 50 + "\n")

    if user_input == "exitai":
        break

    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

    print(result)
