import streamlit as st 
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import load_prompt 
from utils.utils import * 


def set_prompt(keys = '일반 질의응답'):
    if keys == "일반 질의응답":
        prompt = load_prompt(path='prompts/general.yaml', encoding='UTF-8')
    elif keys == "진동 전문가":
        prompt = load_prompt(path='prompts/vibration.yaml', encoding='UTF-8')
    else:  # SNS 전문가
        prompt = load_prompt(path='prompts/sns.yaml', encoding='UTF-8')
    return prompt
    
def write_messeges():
    for chat_message in st.session_state['message']:
        with st.chat_message(chat_message.role):
            st.write(chat_message.content)
            
with st.sidebar:
    st.button('대화 초기화', on_click=clear_chat_history)
    option = st.selectbox(
        "템플릿 한번 골라볼까요?",
        ("일반 질의응답", "진동 전문가", "SNS 전문가" ),
        on_change=clear_chat_history
    )
    st.write(f'{option}으로 설정됐습니다!')
    prompt = set_prompt(option)
    

if 'message' not in st.session_state:
    st.session_state['message'] = []

st.title('반갑꼬리')

mat = ''' 
# 실험용 웹 사이트입니다.
''' 
st.markdown(mat)

write_messeges()    
user_prompt = st.chat_input('말해봐 한번')
llm = set_model()



if user_prompt:
    with st.chat_message('user'):
        st.write(user_prompt)
    chain = prompt | llm 
    answer = chain.stream({'question':user_prompt})
    with st.chat_message('LANI'):
        countainer = st.empty()
        temp = '' 
        for token in answer:
            temp += token.content
            countainer.markdown(temp)
                         
    
    # 대화 기록 저장하기 
    setstate(role='user', content=user_prompt)
    setstate(role='LANI', content=temp)
    
    # st.session_state['message'].append(('user',user_prompt))
    # st.session_state['message'].append(('LANI',user_prompt))
