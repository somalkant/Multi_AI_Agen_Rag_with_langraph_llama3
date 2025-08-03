#!/usr/bin/env python3
"""
Streamlit Web Application for Enhanced Multi-AI Agent RAG System with MCP Agent
"""

import streamlit as st
import asyncio
import os
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_huggingface import HuggingFaceEmbeddings

# LangGraph imports
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

# MCP imports
from mcp_use import MCPClient, MCPAgent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enhanced Multi-AI Agent RAG System with MCP Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .response-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-info {
        color: #17a2b8;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

# AstraDB Configuration
ASTRA_DB_APPLICATION_TOKEN = "xxxx" # Replace with your AstraDB application token
ASTRA_DB_ID = "xxxx" # Replace with your AstraDB database ID


@st.cache_resource
def initialize_system():
    """Initialize the RAG system components"""
    try:
        # Initialize AstraDB
        import cassio
        cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
        
        # Initialize LLM
        GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        if not GROQ_API_KEY:
            st.error("❌ GROQ_API_KEY not found in environment variables!")
            st.info("💡 Please create a .env file with your GROQ_API_KEY")
            return None
        
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize vector store
        astra_vector_store = Cassandra(
            embedding=embeddings,
            table_name="qa_mini_demo",
            session=None,
            keyspace=None
        )
        
        # Initialize retriever
        retriever = astra_vector_store.as_retriever()
        
        # Initialize Wikipedia tool
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
        
        return {
            'llm': llm,
            'embeddings': embeddings,
            'retriever': retriever,
            'wiki': wiki,
            'astra_vector_store': astra_vector_store
        }
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        return None

def initialize_vector_store():
    """Initialize the vector store with sample documents"""
    try:
        with st.spinner("🔄 Initializing vector store with sample documents..."):
            # Docs to index
            urls = [
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
                "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            ]

            # Load
            docs = [WebBaseLoader(url).load() for url in urls]
            docs_list = [item for sublist in docs for item in sublist]

            # Split
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=500, chunk_overlap=0
            )
            doc_splits = text_splitter.split_documents(docs_list)

            # Add to vector store
            system_components = initialize_system()
            if system_components:
                system_components['astra_vector_store'].add_documents(doc_splits)
                st.success(f"✅ Successfully inserted {len(doc_splits)} documents into vector store!")
                return True
    except Exception as e:
        st.error(f"❌ Error initializing vector store: {str(e)}")
        return False

def build_enhanced_graph(system_components):
    """Build the enhanced LangGraph workflow"""
    llm = system_components['llm']
    embeddings = system_components['embeddings']
    retriever = system_components['retriever']
    wiki = system_components['wiki']
    
    # Prompt template for synthetic query generation
    synthetic_query_prompt = PromptTemplate.from_template(
        "Given the following document, generate a question that this document can best answer:\n\n{document_text}\n\nQuestion:"
    )
    
    # Cosine similarity function
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Generate synthetic question for a document
    def get_synthetic_query(document: Document):
        prompt = synthetic_query_prompt.format(document_text=document.page_content)
        response = llm.invoke(prompt)
        return response.content.strip()
    
    # Enhanced routing model
    from pydantic import BaseModel, Field
    from typing import Literal
    
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""
        datasource: Literal["vectorstore", "wiki_search", "mcp_agent"] = Field(
            ...,
            description="Given a user question choose to route it to vectorstore, wikipedia, or MCP agent for real-time web search and tool usage.",
        )
    
    # Enhanced routing prompt
    system = """You are an expert at routing a user question to the most appropriate datasource.

Available datasources:
1. vectorstore: Contains documents related to agents, prompt engineering, and adversarial attacks. Use for questions on these specific topics.
2. wiki_search: Use for general knowledge questions, historical facts, biographies, and topics not covered in the vectorstore.
3. mcp_agent: Use for real-time information needs, current events, live data, web searches, travel/accommodation queries, or when you need to interact with external tools and services.

CRITICAL ROUTING RULES:
- ALWAYS use vectorstore for questions about "agent", "agents", "AI agents", "prompt engineering", "adversarial attacks", "LLM agents", "autonomous agents"
- Use wiki_search for general knowledge questions, historical facts, or biographical information
- Use mcp_agent ONLY for current events, real-time data, travel queries, weather, news, or when you need live web search capabilities

Examples:
- "What is agent?" → vectorstore
- "What are AI agents?" → vectorstore  
- "How do agents work?" → vectorstore
- "Who is Obama?" → wiki_search
- "Current weather in NYC" → mcp_agent
- "Find Airbnb in Paris" → mcp_agent"""
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    
    structured_llm_router = llm.with_structured_output(RouteQuery)
    question_router = route_prompt | structured_llm_router
    
    # Enhanced Graph State
    class GraphState(TypedDict):
        """Represents the state of our enhanced graph."""
        question: str
        generation: str
        documents: List[str]
        mcp_response: str
    
    # HyQE-style retrieve function
    def retrieve(state):
        """Retrieve documents with HyQE reranking using ChatGroq LLaMA 3.3 70B."""
        st.info("🔍 Using Vector Store Agent (HyQE Reranking)")
        question = state["question"]

        # Step 1: Retrieve top-k documents
        initial_docs = retriever.invoke(question)

        # Step 2: Embed the real user query
        question_embedding = embeddings.embed_query(question)

        reranked_docs = []
        for doc in initial_docs:
            try:
                # Generate a synthetic query for this document
                synthetic_q = get_synthetic_query(doc)

                # Embed the synthetic query
                synthetic_q_embedding = embeddings.embed_query(synthetic_q)

                # Score similarity
                similarity = cosine_similarity(question_embedding, synthetic_q_embedding)
                reranked_docs.append((doc, similarity))
            except Exception as e:
                st.warning(f"Warning: Error processing doc: {e}")
                continue

        # Step 3: Sort by similarity
        reranked_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in reranked_docs[:5]]

        return {"documents": top_docs, "question": question}
    
    # Wikipedia search function
    def wiki_search(state):
        """Wikipedia search with HyQE-style reranking."""
        st.info("📚 Using Wikipedia Agent (HyQE Reranking)")
        question = state["question"]

        # Step 1: Wikipedia search
        docs = wiki.invoke({"query": question})

        # Step 2: Wrap if it's just a string or one long block
        if isinstance(docs, str):
            docs = [Document(page_content=docs)]
        elif isinstance(docs, Document):
            docs = [docs]
        elif isinstance(docs, list):
            docs = [Document(page_content=doc) if isinstance(doc, str) else doc for doc in docs]

        # Step 3: Embed original question
        question_embedding = embeddings.embed_query(question)

        reranked_docs = []
        for doc in docs:
            try:
                # Generate a synthetic question the doc can answer
                synthetic_q = get_synthetic_query(doc)
                
                # Embed the synthetic question
                synthetic_q_embedding = embeddings.embed_query(synthetic_q)
                
                # Compute similarity
                similarity = cosine_similarity(question_embedding, synthetic_q_embedding)
                reranked_docs.append((doc, similarity))
            except Exception as e:
                st.warning(f"Warning: Error processing wiki doc: {e}")
                continue

        # Step 4: Sort and select top N
        reranked_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in reranked_docs[:3]]

        return {"documents": top_docs, "question": question}
    
    # MCP Agent function
    def mcp_agent_search(state):
        """MCP Agent search function that uses various MCP tools for real-time information."""
        st.info("🌐 Using MCP Agent (Real-time Tools)")
        question = state["question"]
        
        try:
            # Initialize MCP client and agent
            client = MCPClient.from_config_file("browser_mcp.json")
            agent = MCPAgent(llm=llm, client=client)
            
            # Run the MCP agent asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(agent.run(question))
            loop.close()
            
            return {"mcp_response": response, "question": question}
            
        except Exception as e:
            st.error(f"Error in MCP agent: {e}")
            return {"mcp_response": f"Error: {str(e)}", "question": question}
    
    # Enhanced routing function
    def route_question(state):
        """Enhanced route question to wiki search, RAG, or MCP agent."""
        question = state["question"]
        
        # Debug: Show the question being routed
        st.info(f"🔍 Routing question: '{question}'")
        
        source = question_router.invoke({"question": question})
        
        # Debug: Show the routing decision
        st.info(f"🤖 Router decision: {source.datasource}")
        
        if source.datasource == "wiki_search":
            st.success("🎯 Routing to Wikipedia Search")
            return "wiki_search"
        elif source.datasource == "vectorstore":
            st.success("🎯 Routing to Vector Store")
            return "vectorstore"
        elif source.datasource == "mcp_agent":
            st.success("🎯 Routing to MCP Agent")
            return "mcp_agent"
        else:
            st.warning("⚠️ Defaulting to MCP Agent")
            return "mcp_agent"
    
    # Build the graph
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("mcp_agent", mcp_agent_search)

    # Build graph with enhanced routing
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
            "mcp_agent": "mcp_agent",
        },
    )

    # Add edges to END
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)
    workflow.add_edge("mcp_agent", END)

    return workflow.compile()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Enhanced Multi-AI Agent RAG System with MCP Agent</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            system_components = initialize_system()
            if system_components:
                st.session_state.system_components = system_components
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
        
        # Initialize vector store button
        if st.button("📚 Initialize Vector Store"):
            if initialize_vector_store():
                st.success("✅ Vector store initialized!")
        
        # System status
        st.header("📊 System Status")
        if st.session_state.system_initialized:
            st.markdown('<p class="status-success">✅ System Ready</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⚠️ System Not Initialized</p>', unsafe_allow_html=True)
        
        # Agent information
        st.header("🤖 Available Agents")
        st.markdown("""
        **🔍 Vector Store Agent**
        - Specialized knowledge about AI agents, prompt engineering, and adversarial attacks
        - Uses HyQE reranking for improved relevance
        
        **📚 Wikipedia Agent**
        - General knowledge, historical facts, and biographical information
        - Uses HyQE reranking for improved relevance
        
        **🌐 MCP Agent**
        - Real-time data, current events, web searches, and tool interactions
        - Supports web search, Airbnb search, browser automation
        """)
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
    
    # Main content area
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first using the sidebar button.")
        st.info("💡 The system will automatically route your queries to the most appropriate agent based on the content.")
        return
    
    # Chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "agent_type" in message:
                st.info(f"Agent used: {message['agent_type']}")
            if "documents" in message and message["documents"]:
                with st.expander("📄 Retrieved Documents"):
                    for i, doc in enumerate(message["documents"][:3]):
                        st.write(f"**Document {i+1}:**")
                        st.write(doc.page_content[:300] + "...")
    
    # User input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("🤔 Processing your query..."):
                try:
                    # Build the graph
                    app = build_enhanced_graph(st.session_state.system_components)
                    
                    # Process the query
                    inputs = {"question": prompt}
                    
                    # Create a placeholder for streaming output
                    response_placeholder = st.empty()
                    agent_type_placeholder = st.empty()
                    documents_placeholder = st.empty()
                    
                    response_text = ""
                    agent_type = ""
                    documents = []
                    
                    for output in app.stream(inputs):
                        for key, value in output.items():
                            if key == "retrieve" or key == "wiki_search":
                                if value.get('documents'):
                                    documents = value['documents']
                                    agent_type = "Vector Store" if key == "retrieve" else "Wikipedia"
                            elif key == "mcp_agent":
                                if value.get('mcp_response'):
                                    response_text = value['mcp_response']
                                    agent_type = "MCP Agent"
                    
                    # If no MCP response, generate response from documents
                    if not response_text and documents:
                        llm = st.session_state.system_components['llm']
                        doc_content = "\n\n".join([doc.page_content for doc in documents[:3]])
                        prompt_template = f"""Based on the following documents, please provide a comprehensive answer to the user's question.

Documents:
{doc_content}

User Question: {prompt}

Answer:"""
                        response = llm.invoke(prompt_template)
                        response_text = response.content
                    
                    # Display response
                    response_placeholder.write(response_text)
                    agent_type_placeholder.info(f"🤖 Agent used: {agent_type}")
                    
                    # Display documents if available
                    if documents:
                        with documents_placeholder.expander("📄 Retrieved Documents"):
                            for i, doc in enumerate(documents[:3]):
                                st.write(f"**Document {i+1}:**")
                                st.write(doc.page_content[:300] + "...")
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response_text,
                        "agent_type": agent_type,
                        "documents": documents
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })

if __name__ == "__main__":
    main() 