from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from dotenv import load_dotenv

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
)

chat_model = ChatHuggingFace(llm=llm)

model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    return_full_text=False,
)
local_llm = HuggingFacePipeline(pipeline=pipe)

# local_llm = HuggingFacePipeline.from_model_id(
#     task="text-generation",
#     model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "do_sample": False,
#         "return_full_text": False,
#     },
# )

query = "What is the capital of France?"

# print(chat_model.invoke(query))
# print("\n\n")
response = local_llm.invoke(query)
print(repr(response))
print("\n\n")
query_embeddings = embeddings_model.embed_query(query)
print(f"First 10 dimensions of the query: {query_embeddings[:10]}")
