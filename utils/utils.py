from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import load_prompt 
import streamlit as st 


def set_model():
    llm = ChatOpenAI(
        temperature= 0.2,
        model= 'gpt-4.1-nano-2025-04-14',
    )
    return llm 

def clear_chat_history():
    st.session_state.message = []

def setstate(role,content): 
    return st.session_state['message'].append(ChatMessage(role=role, content= content))
