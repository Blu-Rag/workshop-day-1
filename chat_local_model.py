from local_model_install import chat_model, llm
# from langchain.messages import HumanMessage, SystemMessage

# messages = [
#     SystemMessage(
#         content="You are a helpful assistant."
#     ),
#     HumanMessage(content="Tell me a joke"),
# ]

response = llm.invoke("Write a 3 line poem about AI.")

print(response)
