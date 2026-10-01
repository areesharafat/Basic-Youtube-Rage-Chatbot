import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="assets/youtube.webp",
)

st.title("YouTube RAG Chatbot")
st.write("Ask questions about any YouTube video.")


@st.cache_resource
def load_model():
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-8B",
        task="text-generation",
        max_new_tokens=2048,
        temperature=0.0,
        huggingfacehub_api_token=os.getenv(
            "HUGGINGFACEHUB_ACCESS_TOKEN"
        ),
    )

    return ChatHuggingFace(llm=endpoint)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


model = load_model()
embeddings = load_embeddings()


prompt = PromptTemplate(
    template="""
    You are a helpful assistant.

    Answer only from the PROVIDED TRANSCRIPT.

    If you can't find a relevant answer, just say:
    "YOU DON'T KNOW"

    PROVIDED TRANSCRIPT:
    {context}

    Question:
    {question}
    """,
    input_variables=["context", "question"],
)


youtube_url = st.text_input(
    "YouTube URL",
    placeholder="Paste your YouTube link here..."
)


if st.button("Load Video"):

    if not youtube_url:
        st.warning("Please enter a YouTube URL.")

    else:
        with st.spinner("Loading transcript to answer your queries..."):

            loader = YoutubeLoader.from_youtube_url(youtube_url)
            doc = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(doc)

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.vectorstore = vectorstore

        st.success("Video loaded successfully! 🎉")


question = st.text_input(
    "Ask a question",
    placeholder="What is an LLM?"
)


if st.button("Ask Question"):

    if "vectorstore" not in st.session_state:
        st.warning("Please load a YouTube video first.")

    elif not question:
        st.warning("Please enter a question.")

    else:
        with st.spinner("Thinking..."):

            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            retrieved_docs = retriever.invoke(question)

            combine_docs_chain = create_stuff_documents_chain(
                model,
                prompt
            )

            result = combine_docs_chain.invoke({
                "context": retrieved_docs,
                "question": question
            })

        st.subheader("Answer")

        if hasattr(result, "content"):
            st.write(result.content)
        else:
            st.write(result)