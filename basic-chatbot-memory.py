from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=150,
)
chatbot = ChatHuggingFace(llm=llm)


print("Choose the tone of the chatbot:")
print("""1. Friendly
      2. Professional
      3. Humorous
      4. Custom Instructions""")
choice = input("Choice: ")
if choice == "1":
    mode = "You are a friendly and helpful assistant."
elif choice == "2":
    mode = "You are a professional and knowledgeable assistant."
elif choice == "3":
    mode = "You are a humorous and entertaining assistant."
elif choice == "4":
    mode = input("Enter custom instructions: ")


memory = [SystemMessage(content=mode)]


print("Start chatting with the AI (type 'exit ai' to stop):")
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit ai":
        break
    memory.append(HumanMessage(content=user_input))
    response = chatbot.invoke(memory)
    memory.append(AIMessage(content=response.content))
    print((f"AI: {response.content}"))
