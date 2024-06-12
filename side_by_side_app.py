import streamlit as st
import os
# import chromadb
import streamlit as st
# from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import utils as chromautils

from langchain.retrievers import (
    MergerRetriever,
)

from langchain_pinecone import PineconeVectorStore


# from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere

from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from fake_useragent import UserAgent
from bs4 import BeautifulSoup as Soup

# from link_extractor import LinkScraper
# from web_scraping import DataEngine

# environ for langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-infobell"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


USER_CREDENTIALS = {
    "amd": {"username": "amd", "password": "amd$$1234", "redirect_url": "/amd"},
    "infobellit": {"username": "infobellit", "password": "info$$1234", "redirect_url": "/infobellit"},
    "intel": {"username": "intel", "password": "intel$$1234", "redirect_url": "/intel"},
    "nvidia": {"username": "nvidia", "password": "nvidia$$1234", "redirect_url": "/nvidia"},
}

# Login page
def login_page():
    """
    Displays a login page for user authentication.
    """
    st.title("Login")
    # selected_user = st.selectbox("Select User", list(USER_CREDENTIALS.keys()))
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user_info = USER_CREDENTIALS[username]
        if username == user_info["username"] and password == user_info["password"]:
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.session_state.user = username
            # Set query parameters and rerun the app
            # st.set_query_params(user=user_info["redirect_url"].strip("/"))
            # st.experimental_rerun()

            st.experimental_set_query_params(user=user_info["redirect_url"].strip("/"))
            st.experimental_rerun()

            # st.query_params(user=user_info["redirect_url"].strip("/"))
            # st.rerun()
        else:
            st.error("Invalid username or password")



def check_url(url):
    return url



# Post-processing
def format_docs(docs):
    # st.write(len(docs))
    return "\n\n".join(doc.page_content for doc in docs)


def utf8len(s):
    return len(s.encode('utf-8'))

def process_docs(_docs, web_url, max_depth):
    # st.json(_docs, expanded=False) 
    # docs = chromautils.filter_complex_metadata(_docs)
    byte_max = 40960
    screened_docs = []
    discarded_docs = 0
    for i, doc in enumerate(_docs):
        if utf8len(doc.page_content) <= byte_max:
            screened_docs.append(doc.page_content)
        else:
            discarded_docs += 1
    st.info(f"{discarded_docs} docs have size greater than allowed.")
    return screened_docs, discarded_docs


# @st.cache_resource
def get_vectorstore(_docs, web_url, depth):
    embd = CohereEmbeddings(model="embed-english-v3.0")
    # Chroma vectorstore
    try:
        chroma = Chroma(collection_name="temp_data")
        chroma.delete_collection()
        chroma_vectorstore = Chroma.from_texts(_docs, embd, collection_name="temp_data")
        retriever_chroma = chroma_vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 10}
        )
        # Initialize the BM25 retriever
        bm25_retriever = BM25Retriever.from_texts(_docs)
        bm25_retriever.k = 3

        # Initialize the ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever_chroma],
            weights=[0.4, 0.6]
            )
    except Exception as e:
        st.error(e)
        st.error("Error while creating Embeddings!")
    else:
        return ensemble_retriever

@st.cache_resource
def get_pinecone_vectorstore(index_name):
    # embd = OpenAIEmbeddings(model="text-embedding-3-large")
    embd = CohereEmbeddings(model="embed-english-v2.0")
    # Initialize withour adding records
    vectorstore = PineconeVectorStore(
                    index_name=index_name,
                    embedding=embd
                    )
    # retriever = vectorstore.as_retriever(
	# 	search_type="mmr",
    # 		search_kwargs={'k': 5, 'fetch_k': 10}
	# 	)
    return vectorstore

def upsert_data_pinecone(vectorstore, chunks):
    ids_list = []
    if len(chunks) < 80:
        ids_list = vectorstore.add_texts(chunks)
    else:
        for i in range(0, len(chunks), 100):
            ids_list.extend(vectorstore.add_texts(all_texts[i:i+100]))
            time.sleep(60)
    if len(ids_list):
        st.info("Data Stored Successfully!")
    else:
        st.error("Problem while upserting data!")
    

# @st.cache_resource
def create_chain(_retriever, selected_model):
    model = models[selected_model]
    # st.write(model)
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. 
        \n\nQuestion: {question} \n\nContext: {context}
        """
        )

    # RAG Chain
    # st.write(_retriever)
    rag_chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain

# Main function
if __name__ == "__main__":
    # Set the webpage title
    st.set_page_config(
        page_title="ConvoGene",
        layout="wide",
        initial_sidebar_state='collapsed',
    )
    # Check login status (optional)
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Proceed to chat bot page if logged in, otherwise show login page
    if st.session_state.logged_in:
        user = st.session_state.user
        user_info = USER_CREDENTIALS[user]
        st.experimental_set_query_params(user=user_info["redirect_url"].strip("/"))
        
        # st.set_page_config(layout="wide")
        # st.markdown("""<div id="content">
        #     <img src="infobellLogo.png" class="ribbon" alt="" />
        #     <div>.</div>
        # </div>""", unsafe_allow_html=True)

        header = st.container()
        # header.header(":rainbow[Convogene.ai]", divider='rainbow')
        # Center-align and change the color of the text
        h, image_col = header.columns([0.9, 0.1]) 
        h.markdown("<h2 style='text-align: center; color: black;'>ConvoGene</h2>", unsafe_allow_html=True)
        # header.header("ConvoGene")
        h.caption("<h3 style='text-align: center; color: black;'>Customizable Chat Bot for the Enterprise</h2>", unsafe_allow_html=True)
        # _, image_col = st.columns([0.9, 0.1])
        image_col.image("infobellLogo.png")
        # image_col.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
        header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
        models_header = st.container()
        models_header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
        col1, col2, col3 = header.columns(3)
        col1.text("Cohere")
        # col1.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
        col2.text("OpenAI-gpt4")
        # col2.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
        col3.text("OpenAI-gpt3")
        # col3.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
        ### Custom CSS for the sticky header
        st.markdown(
            """
        <style>
            div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                position: sticky;
                top: 2.875rem;
                # top: 1.5rem;
                background-color: white;
                z-index: 999;
            }
            .fixed-header {
                # border-bottom: 1px solid black;
            }
        </style>
            """,
            unsafe_allow_html=True
        )

        # model list
        models = {
            "OpenAI": ChatOpenAI(
                model="gpt-3.5-turbo-0125",
                temperature=0,
                streaming=True,
                # max_tokens=1000,
            ),
            "OpenAI-GPT4": ChatOpenAI(
                model="gpt-4-turbo",
                temperature=0,
                streaming=True,
                # max_tokens=1000,
            ),
            "Cohere": ChatCohere(
                model="command-r-plus",
                temperature=0,
                streaming=True,
                # max_tokens=1000
            )
        }
        with st.sidebar:
            # Select model using dropdown
            # selected_model = st.selectbox("Select Model", list(models.keys()))
            # st.write("Available Domains: ")
            # # for domain in st.session_state["domains"]["ALL_DATA"]:
            # with st.expander("Data Available: ", expanded=False):
            #     column_config = st.column_config.TextColumn("Domains", width="medium",)
            #     st.dataframe(st.session_state["domains"], column_config=column_config)

            web_url = st.text_input("Enter a url and click on process")
            # max_links = st.number_input("Max No. of links", min_value=1, value=2, placeholder="Max number of links to extract data from")
            depth = [1, 2, 3, 4]
            max_depth = st.selectbox("Select the max depth", depth)
            persist_data = st.toggle("Store Data in Database")
            if st.button("Process"):
                with st.spinner("Processing"):
                    header_template = {}
                    header_template["User-Agent"] = UserAgent().random
                    loader = RecursiveUrlLoader(
                        url=web_url,
                        headers=header_template,
                        max_depth=max_depth,
                        extractor=lambda x: Soup(x, "html.parser").text
                        )
                    docs = loader.load()
                    chunks, discarded_docs = process_docs(docs, web_url, max_depth)
                    st.info(f"No. of Pages Extracted: {len(docs)} - {discarded_docs} = {len(docs) - discarded_docs}")
                    # st.write("Documents Retrieved: ")
                    with st.expander("Retrieved Data:", expanded=False):
                        st.write(chunks,)
                    if len(chunks) > 0:
                        if persist_data:
                            try:
                                vectorstore = get_pinecone_vectorstore(index_name)

                                upsert_data_pinecone(vectorstore, chunks)
                                retriever = vectorstore.as_retriever(
                                    search_type="mmr",
                                    search_kwargs={'k': 5, 'fetch_k': 10}
                                )
                            except Exception as e:
                                print(e)
                                st.error("Error Occured while upserting data!")
                        else:
                            retriever_local = get_vectorstore(chunks, web_url, max_depth)
                            retriever_pinecone = st.session_state["retriever"]
                            retriever = MergerRetriever(retrievers=[retriever_local, retriever_pinecone])
                        if retriever:
                            st.session_state["retriever"] = retriever

        
        # related to Pinecone 
        text_field = "text" # the metadata field that contains our text


        index_name = "cohere-rag-amd"
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = {"cohere": [], "gpt3": [], "gpt4": []}

        if "curr_url" not in st.session_state:
            st.session_state["curr_url"] = ""

        if "retriever" not in st.session_state:
            st.session_state["retriever"] = get_pinecone_vectorstore(index_name).as_retriever()

        retriever = st.session_state["retriever"]
        # st.write(retriever)
        rag_chain_cohere = create_chain(retriever, "Cohere")
        rag_chain_gpt4 = create_chain(retriever, "OpenAI-GPT4")
        rag_chain_gpt3 = create_chain(retriever, "OpenAI")
        
        # model_container = st.container(border=True)
        cohere, gpt4, gpt3  = st.columns(3)
        cohere, gpt4, gpt3 = cohere.container(border=True), gpt4.container(border=True), gpt3.container(border=True)
        with cohere:
            # st.text("Cohere")
            # st.image("https://static.streamlit.io/examples/cat.jpg")
            for message in st.session_state.messages["cohere"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        with gpt4:
            # st.text("OpenAI-GPT4")
            # st.image("https://static.streamlit.io/examples/owl.jpg")
            for message in st.session_state.messages["gpt4"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        with gpt3:
            # st.text("OpenAI-GPT3")
            # st.image("https://static.streamlit.io/examples/dog.jpg")
            for message in st.session_state.messages["gpt3"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Enter your question here"):

            # Display user message in chat message container
            with cohere.chat_message("user"):
                st.markdown(prompt)

            with gpt4.chat_message("user"):
                st.markdown(prompt)

            with gpt3.chat_message("user"):
                st.markdown(prompt)


            # Display assistant response in chat message container
            with cohere.chat_message("assistant"):
                try:
                    stream = rag_chain_cohere.stream(prompt)
                    response_cohere = st.write_stream(stream)
                    # Add user message to chat history
                    st.session_state.messages["cohere"].append({"role": "user", "content": prompt})
                    # Add response to chat history
                    st.session_state.messages["cohere"].append({"role": "assistant", "content": response_cohere})

                except Exception as e:
                    st.exception(f"Error Occured: {str(e)}")
            
            with gpt4.chat_message("assistant"):
                try:
                    stream = rag_chain_gpt4.stream(prompt)
                    response_gpt4 = st.write_stream(stream) 
                    # Add user message to chat history
                    st.session_state.messages["gpt4"].append({"role": "user", "content": prompt})
                    # Add response to chat history
                    st.session_state.messages["gpt4"].append({"role": "assistant", "content": response_gpt4})

                except Exception as e:
                    st.exception(f"Error Occured: {str(e)}")
            
            with gpt3.chat_message("assistant"):
                try:
                    stream = rag_chain_gpt3.stream(prompt)
                    response_gpt3 = st.write_stream(stream)
                    # Add user message to chat history
                    st.session_state.messages["gpt3"].append({"role": "user", "content": prompt})
                    # Add response to chat history
                    st.session_state.messages["gpt3"].append({"role": "assistant", "content": response_gpt3})

                except Exception as e:
                    st.exception(f"Error Occured: {str(e)}")

            
            # st.write_stream(stream)
    else:
        login_page()
        
