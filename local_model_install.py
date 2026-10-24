from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()

hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFacePipeline.from_model_id(
    model_id="sapientinc/HRM-Text-1B",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.8, max_new_tokens=150, top_p=0.8, repetition_penalty=1.4
    ),
)
chat_model = ChatHuggingFace(llm=llm)
