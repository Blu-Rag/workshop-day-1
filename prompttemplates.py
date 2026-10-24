from langchain_core.prompts import ChatPromptTemplate

web_search_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that gather the information from the web."),
    ("user", "Find the information about {topic}."),
])

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured and insightful report.",
    ),
    (
        "user",
        """Write a detailed report on the topic below.

     Topic: {topic}

     Research Gathered: {research}
     """,
    ),
])

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest, unbiased and specific in your feedback.",
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

     Report: {report}

     Respond in this exact format:

     Score: X/10

     Strenghts:
     --...
     --...

     Areas to Improve:
     --...
     --...

     One line verdict:
     ...""",
    ),
])

__all__ = ["critic_prompt", "web_search_prompt", "writer_prompt"]
