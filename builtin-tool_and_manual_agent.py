import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from rich import print

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

    params = {"q": city, "units": "metric"}

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

tools = {"get_weather": get_weather, "tavily_search": tavily_search}

llm_with_tool = llm.bind_tools([get_weather, tavily_search])

# Agent Loop - Very Important

messages = []
print("City Intelligence System")
print("Type Exitai to exit")
print("\n" + "=" * 50 + "\n")

# str_output = StrOutputParser()

while True:
    user_input = input("You: ")
    print("\n" + "=" * 50 + "\n")
    if user_input.lower() == "exitai":
        break
    messages.append(HumanMessage(user_input))

    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        # if tool is required

        if result.tool_calls:
            for i in result.tool_calls:
                tool_name = i["name"]
                # tool execution
                tool_result = tools[tool_name].invoke(i)
                messages.append(
                    tool_result
                )  # by defualt tool call is an object of ToolMessage containing id of tool also

            continue
        else:
            print(f"AI: {result.content}")
            print("\n" + "=" * 50 + "\n")
            break
