import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph


# --- Page Configuration ---
st.set_page_config(
    page_title="Smart RAG Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIRECTORY = "chroma_db"
LLM_MODEL_ID = "openai/gpt-oss-20b"


# --- Embeddings (cached) ---
@st.cache_resource
def get_embeddings():
    """Returns a cached HuggingFace embeddings instance."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- RAG Functions ---
def ingest_uploaded_pdfs(uploaded_files):
    """Ingests uploaded PDF files into Chroma VectorDB."""
    documents = []

    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file to a temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            # Tag each document with the original filename
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            documents.extend(docs)

            # Clean up temp file
            os.unlink(tmp_path)
        except Exception as e:
            st.warning(f"Error loading {uploaded_file.name}: {e}")

    if not documents:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    embeddings = get_embeddings()

    # Append to existing vectorstore or create new one
    if os.path.exists(PERSIST_DIRECTORY):
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        vectorstore.add_documents(texts)
    else:
        vectorstore = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY
        )

    return len(documents)


def create_retriever():
    """Creates a retriever from the Chroma DB."""
    if not os.path.exists(PERSIST_DIRECTORY):
        return None
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    # Check if there are any documents
    if vectorstore._collection.count() == 0:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# --- Graph State Definition ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    sender: str
    answer: str


# --- Graph Nodes ---
def router_node(state: GraphState) -> str:
    """The Router. Decides which tool to use next."""
    question = state["question"]

    has_vectorstore = create_retriever() is not None

    if not has_vectorstore:
        return "web_search"

    routing_prompt = f"""You are an expert at routing user questions.
    The **vectorstore** contains user-uploaded documents (PDFs). Use it when the question likely relates to the content of those documents.
    The **web_search** tool can access real-time information from the internet. Use it for current events, latest news, or topics not covered in the uploaded documents.
    Use **both** when the question could benefit from combining document knowledge with web information.

    Based on the user's question, decide the best approach.
    Question: "{question}"
    Respond with only one of: 'vectorstore', 'web_search', or 'both'."""

    llm = ChatGroq(temperature=0, model_name=LLM_MODEL_ID)
    response = llm.invoke(routing_prompt)
    decision = response.content.strip().lower()

    if "both" in decision:
        return "both"
    elif "web_search" in decision:
        return "web_search"
    else:
        return "vectorstore"


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieves documents from the vectorstore."""
    question = state["question"]
    retriever = create_retriever()
    if retriever is None:
        return {"documents": [], "sender": "retrieve_node"}
    retrieved_docs = retriever.invoke(question)
    return {"documents": retrieved_docs, "sender": "retrieve_node"}


def web_search_node(state: GraphState) -> GraphState:
    """Searches the web for information."""
    question = state["question"]

    tavily_search = TavilySearch(
        max_results=3,
        search_depth="advanced"
    )
    search_results = tavily_search.invoke(question)

    web_docs = []
    if isinstance(search_results, list):
        for result in search_results:
            content = result.get("content", "")
            url = result.get("url", "web")
            if content:
                web_docs.append(Document(
                    page_content=content,
                    metadata={"source": url}
                ))
    elif isinstance(search_results, dict) and "results" in search_results:
        for result in search_results["results"]:
            content = result.get("content", "")
            url = result.get("url", "web")
            if content:
                web_docs.append(Document(
                    page_content=content,
                    metadata={"source": url}
                ))

    return {"documents": web_docs, "sender": "web_search_node"}


def combined_node(state: GraphState) -> GraphState:
    """Retrieves from both vectorstore and web search, merges results."""
    question = state["question"]

    # Get vectorstore results
    all_docs = []
    retriever = create_retriever()
    if retriever is not None:
        retrieved_docs = retriever.invoke(question)
        for doc in retrieved_docs:
            doc.metadata["retrieval_source"] = "📄 PDF"
        all_docs.extend(retrieved_docs)

    # Get web results
    tavily_search = TavilySearch(
        max_results=2,
        search_depth="advanced"
    )
    search_results = tavily_search.invoke(question)

    if isinstance(search_results, list):
        for result in search_results:
            content = result.get("content", "")
            url = result.get("url", "web")
            if content:
                all_docs.append(Document(
                    page_content=content,
                    metadata={"source": url, "retrieval_source": "🌐 Web"}
                ))
    elif isinstance(search_results, dict) and "results" in search_results:
        for result in search_results["results"]:
            content = result.get("content", "")
            url = result.get("url", "web")
            if content:
                all_docs.append(Document(
                    page_content=content,
                    metadata={"source": url, "retrieval_source": "🌐 Web"}
                ))

    return {"documents": all_docs, "sender": "combined_node"}


def generate_node(state: GraphState) -> GraphState:
    """Generates an answer using the LLM."""
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = f"""You are an expert Q&A assistant. Use the following context to answer the user's question.
If the context does not contain the answer, state that you cannot find the information.
Be concise, helpful, and well-structured in your response. Use markdown formatting where appropriate.

Context:
{context}

Question:
{question}

Answer:"""
    llm = ChatGroq(temperature=0, model_name=LLM_MODEL_ID)
    response = llm.invoke(prompt)
    answer = response.content
    return {"answer": answer, "sender": "generate_node"}


# --- Build the Graph ---
def build_graph():
    """Builds and compiles the workflow graph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("combined", combined_node)
    workflow.add_node("generate", generate_node)

    workflow.add_conditional_edges(
        START,
        router_node,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search",
            "both": "combined",
        },
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("combined", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# --- Streamlit UI ---
def main():
    st.title("🤖 Smart RAG Agent")
    st.markdown("Upload documents & ask anything — powered by AI with web search!")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files_list" not in st.session_state:
        st.session_state.uploaded_files_list = []

    # Sidebar
    with st.sidebar:
        st.header("📤 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload any PDF documents to build your knowledge base"
        )

        if uploaded_files:
            # Check for new files that haven't been ingested yet
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.uploaded_files_list
            ]

            if new_files:
                with st.spinner(f"Ingesting {len(new_files)} PDF(s)..."):
                    doc_count = ingest_uploaded_pdfs(new_files)
                    if doc_count > 0:
                        for f in new_files:
                            st.session_state.uploaded_files_list.append(f.name)
                        st.success(f"✅ Ingested {doc_count} pages from {len(new_files)} file(s)!")
                    else:
                        st.warning("No content found in uploaded files")

        # Show uploaded documents
        if st.session_state.uploaded_files_list:
            st.divider()
            st.subheader("📚 Knowledge Base")
            for fname in st.session_state.uploaded_files_list:
                st.text(f"📄 {fname}")

            if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
                import shutil
                if os.path.exists(PERSIST_DIRECTORY):
                    shutil.rmtree(PERSIST_DIRECTORY)
                st.session_state.uploaded_files_list = []
                st.rerun()

        st.divider()

        # Database Status
        st.subheader("💾 Status")
        if os.path.exists(PERSIST_DIRECTORY):
            retriever = create_retriever()
            if retriever is not None:
                st.success("✅ Knowledge base ready")
            else:
                st.info("📭 Knowledge base is empty")
        else:
            st.info("📭 No documents uploaded yet")

        st.divider()

        # About
        st.subheader("ℹ️ How it works")
        st.markdown("""
        - **📤 Upload** any PDF documents
        - **💬 Ask** questions about your docs
        - **🌐 Web search** for real-time info
        - **🔀 Auto-routes** to the best source
        - **🔗 Combines** sources when needed
        """)

    # Main Chat Interface
    st.divider()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source" in message:
                with st.expander("📚 Sources"):
                    st.markdown(message["source"])

    # Chat input
    if question := st.chat_input("Ask anything..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    app = build_graph()
                    inputs = {"question": question}

                    result = None
                    for output in app.stream(inputs, stream_mode="values"):
                        result = output

                    if result and "answer" in result:
                        answer = result["answer"]

                        # Show routing info
                        sender = result.get("sender", "unknown")
                        if sender == "retrieve_node":
                            st.caption("📄 *Answered from uploaded documents*")
                        elif sender == "web_search_node":
                            st.caption("🌐 *Answered from web search*")
                        elif sender == "combined_node":
                            st.caption("🔗 *Answered from documents + web search*")

                        st.markdown(answer)

                        # Show source information
                        source_text = ""
                        if "documents" in result and result["documents"]:
                            sources = []
                            for doc in result["documents"]:
                                src = doc.metadata.get("source", "")
                                retrieval_src = doc.metadata.get("retrieval_source", "")
                                if src:
                                    label = f"{retrieval_src} {src}" if retrieval_src else src
                                    if label not in sources:
                                        sources.append(label)

                            if sources:
                                source_text = "**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
                                with st.expander("📚 View Sources"):
                                    st.markdown(source_text)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            **({"source": source_text} if source_text else {})
                        })
                    else:
                        error_msg = "I couldn't generate an answer. Please try again."
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()