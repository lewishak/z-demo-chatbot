__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

openai_api_key = st.secrets.OPENAI_API_KEY

def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

    st.title("_Private Data :red[QA Chat]_ :books:")

    # 이미 저장된 백터스토어 가져오기
    dir_path = "./chroma_db/질링스 기업 4000"

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "disabled_text_area" not in st.session_state:
        st.session_state.disabled_text_area = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

        st.markdown("챗 구성중...From " + dir_path)

        context = """- 유사업체, 경쟁사는 추천할 때는 "한줄소개"와 "산업분야" 정보를 비교해서 추천해주세요. 그리고 추천 알고리즘을 설명하지 않고 결과만 바로 보여주세요.
- 결과 표시는 "기업명 :  |한줄소개: | 산업분야: | 투자금 : | 시도: | 설립일 : | 홈페이지 : | 직원수 : " 형식으로 보여주세요.(최대 5개).
- 결과 마지막은 "최대 5개 기업만 표시합니다. 더 많은 중소기업을 찾고 싶으시다면 https://app.zillinks.com/search 에 접속하여 검색하세요. 정보에 관련된 제안이 있다면 info@zillinks.com로 연락주세요." 라는 문구를 띄어줘"""

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", context),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        llm = ChatOpenAI(
                openai_api_key=openai_api_key, 
                # model_name = 'gpt-4',
                model_name = 'gpt-3.5-turbo',
                temperature=0
            )

        vector_store = get_vectorstore(dir_path)

        retriever = vector_store.as_retriever(search_type = 'mmr', vervose = True)

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type= "stuff",
                retriever=retriever,
                condense_question_prompt=chat_prompt,
                memory=memory,
                get_chat_history=lambda h: h,
                return_source_documents=True, # 참고한 소스 문서를 같이 반환하도록 설정
                verbose = True # 상세 모드에서 실행할지 여부입니다. 상세 모드에서는 일부 중간 로그가 콘솔에 인쇄됩니다.
            )

        st.session_state.processComplete = True
        st.session_state.disabled_text_area = True
        st.rerun()

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    #* 메시지 출력 영역
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        #* 사용자 질문을 메시지 출력에 추가
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # 대화 함수 chain 에 할당
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                # chain 함수에 쿼리 전달
                result = chain({"question": query})
                
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']

                response = result['answer']
                st.markdown(response)

                # with st.expander("참고 문서 확인"):
                #     source_documents = result['source_documents']
                #     st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                #     st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                #     st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)


        #* 어시스턴트 답변을 메시지 출력에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})


@st.cache_resource
def get_vectorstore(dir_path):
    # 임베딩 모델
    model_name = "jhgan/ko-sbert-nli"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return Chroma(persist_directory=dir_path, embedding_function=hf)
        

# def time_convert(sec):
#     mins = sec // 60
#     sec = sec % 60
#     hours = mins // 60
#     mins = mins % 60
#     return "Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec)


if __name__ == '__main__':
    main()