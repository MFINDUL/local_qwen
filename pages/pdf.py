import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st 
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import load_prompt 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.utils import * 
import tempfile
import os 

def retrievers_pages(docs):
    temp = '\n\n'
    for i in docs:
        temp += i.page_content
    return temp 
# 환경변수 로드
load_dotenv()

def set_model(option='basic'):
    """기조 set_model 함수를 오버라이딩하여 모델 선택 기능 추가"""
    if option == 'gpt4o':
        llm = ChatOpenAI(
            temperature=0.2,
            model='gpt-4o',
        )
    elif option == 'gpt-o3':
        llm = ChatOpenAI(
            temperature=0.1,
            model='o3-mini',
        )
    else:  # basic
        llm = ChatOpenAI(
            temperature=0.2,
            model='gpt-4o-mini',
        )
    return llm

if 'message' not in st.session_state:
    st.session_state['message'] = []

# @st.cache_resource(show_spinner='파일 업로드 중입니다.')
# def embed_file(file):
#     content = file.read()
#     file_paths = f'./.cache/files/{file.name}'
#     with open(file_paths, 'wb') as f:
#         f.write(content)
      
with st.sidebar:
    st.button('대화 초기화', on_click=clear_chat_history)
    option = st.selectbox(
        "모델을 한번 골라볼까요?",
        ("gpt4o", "basic", "gpt-o3" ),
        on_change=clear_chat_history
    )
    st.write(f'{option}으로 설정됐습니다!')
    

st.header('업로드 한 소설 속 내용 중 궁금한 것을 물어보세요!')
# 업로드 받고 
uploaded_file = st.file_uploader("PDF 파일을 선택하세요")
if uploaded_file:
    # 받은거 읽어오고
    tempdir = tempfile.TemporaryDirectory()
    st.write(tempdir)
    temp_files = os.path.join(tempdir.name, uploaded_file.name)
    with open(temp_files, 'wb') as f:
        f.write(uploaded_file.getvalue())
    pdfs = PyMuPDFLoader(temp_files)
    docs = pdfs.load()
    #받은걸 잘라야함 
    text_splits = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    split_docs = text_splits.split_documents(docs)

    # 임베딩할거임 
    embeddings = OpenAIEmbeddings()
    vectordb= FAISS.from_documents(documents = split_docs, embedding = embeddings)

    # 문맥에 맞게 문서 들고와줄 리트리버 선언 
    retriever = vectordb.as_retriever()

    llm = set_model(option)
    # 모델 설정, 로컬로 바꿀 예정임 
    prompt = load_prompt(r'C:\Users\roaco\Desktop\프로젝트들 정리하기\github\prompts\rags.yaml',encoding='UTF-8')    
    # 프롬프트 맥임 
    user_prompts = st.text_input('예시: 아내는 왜 설렁탕을 먹지 못했나요? ')
    # 사용자 질문 할당 
    if user_prompts:
        with st.chat_message('user'):
            st.write(user_prompts)
        chain =( {'context':retriever | retrievers_pages , 'question':RunnablePassthrough()} |
        prompt | 
        llm 
        )
        # 체인 묶기, {}로 묶인 모든 값에 input이 들어갈 것 , 이후 순서대로 값을 던짐 
        response = chain.stream(user_prompts)
        
        with st.chat_message('LANI'):
            temp = ''
            container = st.empty()
            for token in response:            
                temp += token.content
                container.markdown(temp)
        # 대화 기록 저장하기 
        setstate(role='user', content=user_prompts)
        setstate(role='LANI', content=temp)