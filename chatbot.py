from langchain.chains import RetrievalQA
from langchain_community.llms import huggingface_pipeline
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone
from transformers import pipeline
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

os.environ['PINECONE_API_KEY'] = ""



embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


vector_store = Pinecone.from_existing_index("hello", embedding_model)

model_path= "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# Load Mistral model and tokenizer
# model_id = "mistralai/Mistral-7B-v0.1"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, torch_dtype=torch.float16, device_map="auto"
# )

# Set up the Hugging Face pipeline
""" mistral_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
)

# Wrap in LangChain LLM
mistral_llm = huggingface_pipeline(pipeline=mistral_pipeline) """

mistral_llm = LlamaCpp(
    model_path=model_path,
    temperature=0.3,
    max_tokens=512,
    n_ctx=4096, 
    verbose=True
)

prompt_template = """You are an AI policy expert providing clear and concise answers about AI regulations. 
Use only the provided context to answer the question.

Context:
{context}

Question: {question}

Answer in a professional and factual manner.
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


qa = RetrievalQA.from_chain_type(
    llm=mistral_llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt}, 
)


print("Chatbot is ready. You can now ask questions.")



while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("Goodbye!")
        break
    
    result = qa.invoke(query)
    print("\nChatbot:", result["result"])

