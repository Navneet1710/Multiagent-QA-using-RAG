import sys
import os
import time
import chromadb
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

# Load environment variables from .env file
load_dotenv()
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import SecretStr, BaseModel

# Custom embedding model using Sentence-Transformers
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

# Document processing functions
def load_documents(file_paths: List[str]) -> List[Document]:
    """Load documents from text files."""
    documents = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append(Document(page_content=text, metadata={"source": file_path}))
    return documents

def chunk_documents(documents: List[Document], chunk_size=500, chunk_overlap=50) -> List[Document]:
    """Split documents into chunks of specified size."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def index_documents(documents: List[Document]) -> Chroma:
    """Create vector store from documents."""
    import os
    import shutil

    # Create a persistent directory for ChromaDB
    persist_directory = os.path.join(os.getcwd(), "chroma_db")

    # If the directory exists, remove it to start fresh
    if os.path.exists(persist_directory):
        print(f"Removing existing ChromaDB directory: {persist_directory}")
        shutil.rmtree(persist_directory)

    # Create the directory
    os.makedirs(persist_directory, exist_ok=True)
    print(f"Created ChromaDB directory: {persist_directory}")

    # Initialize the embedding model
    embedding_model = SentenceTransformerEmbeddings()

    # Create the vector store with the persistent directory
    print(f"Creating new vector store with collection name 'rag_qa_docs'")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name="rag_qa_docs",
        persist_directory=persist_directory
    )

    print(f"Vector store created successfully with {len(documents)} documents")
    return vectorstore

def retrieve_relevant_chunks(vectorstore: Chroma, query: str, k=3) -> List[Document]:
    """Retrieve the top-k most similar chunks to the query."""
    docs = vectorstore.similarity_search(query, k=k)
    return docs

# Tool functions
def use_calculator(query: str) -> str:
    """Simple calculator function using Python's eval."""
    # Extract numerical expression from query
    # This is a simplified approach - a real system would need more robust parsing
    query = query.lower()

    # Remove common phrases that might appear in calculation requests
    for phrase in ["calculate", "what is", "compute", "result of", "equals", "equal to"]:
        query = query.replace(phrase, "")

    # Keep only the mathematical expression
    expression = query.strip()
    try:
        result = eval(expression)
        return f"Calculator result: {expression} = {result}"
    except:
        return f"Sorry, I couldn't parse '{query}' as a mathematical expression."

def use_dictionary(query: str) -> str:
    """Simple dictionary lookup function using an external API."""
    # Extract the word to define
    query = query.lower()
    for phrase in ["define", "what is", "what's", "meaning of", "definition of"]:
        query = query.replace(phrase, "")

    word = query.strip()
    try:
        # Using Free Dictionary API
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                meanings = data[0].get("meanings", [])
                if meanings:
                    definition = meanings[0].get("definitions", [{}])[0].get("definition", "No definition found")
                    return f"Definition of '{word}': {definition}"

        return f"Sorry, I couldn't find a definition for '{word}'."
    except:
        return f"Sorry, I couldn't access the dictionary service for '{word}'."

def use_conversion(query: str) -> str:
    """Simple unit conversion function."""
    # This is a very basic implementation
    query = query.lower()

    # Handle some common conversions
    if "miles to kilometers" in query or "miles to km" in query:
        try:
            # Extract the number
            import re
            match = re.search(r'(\d+\.?\d*)', query)
            if match:
                miles = float(match.group(1))
                km = miles * 1.60934
                return f"{miles} miles is approximately {km:.2f} kilometers."
        except:
            pass
    elif "kilometers to miles" in query or "km to miles" in query:
        try:
            import re
            match = re.search(r'(\d+\.?\d*)', query)
            if match:
                km = float(match.group(1))
                miles = km / 1.60934
                return f"{km} kilometers is approximately {miles:.2f} miles."
        except:
            pass

    return f"Sorry, I couldn't perform the conversion for '{query}'. I support basic conversions like miles to kilometers."

# RAG Agent setup
def setup_rag_agent(vectorstore: Chroma):
    """Set up the RAG agent with retrieval and LLM."""

    # Initialize LLM with Groq
    api_key_value = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
        api_key=SecretStr(api_key_value) if api_key_value is not None else None
    )

    # Define a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create a prompt template
    prompt_template = """
    You are a helpful assistant answering questions based on the provided context.

    Context information from relevant documents:
    {context}

    Human Question: {question}

    Instructions:
    1. Answer the question based primarily on the information in the context.
    2. If the context doesn't contain relevant information, say so rather than making up an answer.
    3. Keep your answer concise and to the point.
    4. Cite the relevant parts of the context if appropriate.

    Your answer:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create the RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Agent decision logic
def is_tool_query(query: str) -> bool:
    """Determine if a query should be handled by tools."""
    tool_keywords = ["calculate", "compute", "define", "meaning", "definition", "convert", "conversion"]
    return any(keyword in query.lower() for keyword in tool_keywords)

def get_tool_result(query: str) -> str:
    """Route query to appropriate tool."""
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in ["calculate", "compute"]):
        return use_calculator(query)
    elif any(keyword in query_lower for keyword in ["define", "meaning", "definition"]):
        return use_dictionary(query)
    elif any(keyword in query_lower for keyword in ["convert", "conversion"]):
        return use_conversion(query)
    else:
        return "No appropriate tool found for this query."

# Define the state schema for LangGraph
class AgentState(BaseModel):
    """State for the agent workflow."""
    query: str
    agent_path: str = ""
    retrieved_chunks: List[Dict] = []
    llm_prompt: str = ""
    response: str = ""

# LangGraph workflow setup
def create_agent_workflow(rag_agent, vectorstore):
    """Create the LangGraph workflow with RAG and Tool agents."""

    # Define the nodes in the graph
    def route_query(state):
        """Determine which agent to use."""
        # Router nodes should return a dict with the next node in the 'agent_path' field
        if is_tool_query(state.query):
            return {"agent_path": "tool_agent"}
        else:
            return {"agent_path": "rag_agent"}

    def run_tool_agent(state):
        """Execute the tool agent."""
        result = get_tool_result(state.query)
        return {"agent_path": "tool_agent", "response": result}

    def run_rag_agent(state):
        """Execute the RAG agent."""
        # Get relevant chunks
        query = state.query
        chunks = retrieve_relevant_chunks(vectorstore, query)

        # Format chunks for logging
        chunks_info = []
        for i, chunk in enumerate(chunks):
            chunks_info.append({
                "chunk_id": i+1,
                "content": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
                "source": chunk.metadata.get("source", "unknown")
            })

        # Create prompt for LLM
        context = "\n\n".join(doc.page_content for doc in chunks)
        prompt_text = f"""
        You are a helpful assistant answering questions based on the provided context.

        Context information from relevant documents:
        {context}

        Human Question: {query}

        Instructions:
        1. Answer the question based primarily on the information in the context.
        2. If the context doesn't contain relevant information, say so rather than making up an answer.
        3. Keep your answer concise and to the point.
        4. Cite the relevant parts of the context if appropriate.

        Your answer:
        """

        # Get response from RAG agent
        response = rag_agent.invoke(query)

        return {
            "agent_path": "rag_agent",
            "retrieved_chunks": chunks_info,
            "llm_prompt": prompt_text,
            "response": response
        }

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", route_query)
    workflow.add_node("tool_agent", run_tool_agent)
    workflow.add_node("rag_agent", run_rag_agent)

    # Add conditional edges from router node based on agent_path
    workflow.add_conditional_edges(
        "router",
        lambda state: state.agent_path,  # Use the agent_path field to determine the next node
        {
            "tool_agent": "tool_agent",
            "rag_agent": "rag_agent"
        }
    )

    # Add edges to END
    workflow.add_edge("tool_agent", END)
    workflow.add_edge("rag_agent", END)

    # Set the entry point
    workflow.set_entry_point("router")

    return workflow.compile()

# Create a class to manage the entire system
class RAGQASystem:
    def __init__(self, document_paths):
        # Initialize system components
        if not document_paths:
            raise ValueError("No document paths provided. Please select at least one document.")

        print("Loading documents...")
        self.documents = load_documents(document_paths)

        if not self.documents:
            raise ValueError("No documents loaded. Please check if the document files exist.")

        print("Chunking documents...")
        self.chunked_docs = chunk_documents(self.documents)

        if not self.chunked_docs:
            raise ValueError("No document chunks created. Please check the document content.")

        print("Indexing documents...")
        try:
            self.vectorstore = index_documents(self.chunked_docs)
        except Exception as e:
            raise ValueError(f"Error creating vector store: {str(e)}")

        print("Setting up RAG agent...")
        try:
            self.rag_agent = setup_rag_agent(self.vectorstore)
        except Exception as e:
            raise ValueError(f"Error setting up RAG agent: {str(e)}")

        print("Creating agent workflow...")
        try:
            self.agent_workflow = create_agent_workflow(self.rag_agent, self.vectorstore)
        except Exception as e:
            raise ValueError(f"Error creating agent workflow: {str(e)}")

        print("System initialization complete!")

    def query(self, user_query: str) -> Dict:
        """Process a user query through the agent workflow."""
        start_time = time.time()

        # Create an instance of AgentState with the user query
        input_state = AgentState(query=user_query)

        # Invoke the workflow with the AgentState instance
        result = self.agent_workflow.invoke(input_state)

        # Add processing time to result
        result["processing_time"] = f"{time.time() - start_time:.2f} seconds"

        return result
