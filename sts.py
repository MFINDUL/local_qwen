import streamlit as st 
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import load_prompt 

def set_prompt(keys = '일반 질의응답',paths = 'prompts/general.yaml'):
    if keys == "일반 질의응답":
        # prompt = ChatPromptTemplate.from_messages(
        # [
        #     ('system','당신은 친절한 도우미 LANI입니다. '),
        #     ('user','{question}에 대해 설명해주세요')
        # ]
        prompt = load_prompt(path=paths,encoding='UTF-8')
        # )
    elif keys == "진동 전문가":
        # prompt = ChatPromptTemplate.from_messages(
        # [
        #     ('system','당신은 진동 고장 진단 전문가 LANI입니다.  '),
        #     ('user','{question}에 대해 설명해주세요')
        # ]
        # )
        prompt = load_prompt(path=paths,encoding='UTF-8')
    else:
        # prompt = ChatPromptTemplate.from_messages(
        # [
        #     ('system','당신은 fps 전문가 LANI입니다. '),
        #     ('user','{question}에 대해 설명해주세요')
        # ]
        # )
        prompt = load_prompt(path=paths,encoding='UTF-8')   
    return prompt
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
def write_messeges():
    for chat_message in st.session_state['message']:
        with st.chat_message(chat_message.role):
            st.write(chat_message.content)
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
