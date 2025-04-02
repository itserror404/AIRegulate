import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch

torch.classes.__path__ = [] 

os.environ['PINECONE_API_KEY'] = ""

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

vector_store = Pinecone.from_existing_index("hello", embedding_model)


model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
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


st.title("AIRegulate Chatbot")
st.write("Ask about AI regulations and policies!")


query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query:
        with st.spinner("Generating response..."):
            result = qa.invoke(query)
            st.write("### Answer:")
            st.write(result["result"])
    else:
        st.warning("Please enter a question.")
