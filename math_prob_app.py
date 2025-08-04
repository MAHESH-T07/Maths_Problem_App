import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

st.title("Chatbot that solve's Maths Problems")
groq_api=st.sidebar.text_input("Enter groq Api",type="password")
if groq_api:

    llm = ChatGroq(model='gemma2-9b-it',api_key=groq_api)

    math_prompt = """
    Solve the following math problem step-by-step, showing clear reasoning and calculations. 
    If the problem is ambiguous or lacks enough information, state the assumptions you are making.

    Problem: {text}

    Answer:
    """

    #initialize the tool
    wikipedia_wrapper = WikipediaAPIWrapper()
    wiki_tool = Tool(
        name="wikipedia",
        func=wikipedia_wrapper.run,
        description="It is a tool used to search answers in internet"
    )

    #initialize maths tool
    maths_chain = LLMMathChain.from_llm(llm)
    maths_tool = Tool(
        name="calculator",
        func=maths_chain.run,
        description="It is tool to answer maths related questions only"
    )

    prompt = PromptTemplate(input_variables=["text"],template=math_prompt)

    #initialize reasoning tool
    llm_chain = LLMChain(llm=llm,prompt=prompt)
    reasoning_tool = Tool(
        name="reasoning",
        func=llm_chain.run,
        description="A tool that answer reasoning questions and logical questions"
    )

    #combine all tools
    chain = initialize_agent(
        llm=llm,
        tools=[wiki_tool,maths_tool,reasoning_tool],
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    ## build streamlit ui
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"Assistant","content":"Hi I am Maths Assistant how can i hekp you"}]


    question = st.chat_input("Ask me anything")
    if question:
        st.session_state.messages.append({"role":"user","content":question})
        
        with st.chat_message("Assistant"):
            call_backs = StreamlitCallbackHandler(st.container())
            response = chain.run(question,callbacks=[call_backs])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write(response)