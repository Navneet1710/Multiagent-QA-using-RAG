import os
import streamlit as st
from dotenv import load_dotenv
from utils import RAGQASystem

# Load environment variables from .env file
load_dotenv()

# Debug: Print the API key (first few characters only for security)
api_key = os.environ.get("GROQ_API_KEY")
if api_key:
    print(f"GROQ_API_KEY found: {api_key[:10]}...")
else:
    print("GROQ_API_KEY not found in environment variables!")

# Print all environment variables (excluding their values for security)
print("Environment variables:")
for key in os.environ:
    print(f"  - {key}")

def create_streamlit_app():
    st.set_page_config(
        page_title="RAG Q&A Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("RAG-Powered Multi-Agent Q&A Assistant")

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.system = None

    # Sidebar for document upload and system initialization
    with st.sidebar:
        st.header("System Configuration")

        # Check for API key
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("⚠️ GROQ_API_KEY not found in environment variables!")
            st.info("Please set it before running the application.")

        # Sample document options
        st.subheader("Sample Documents")
        sample_docs = [
            "document1.txt",
            "document2.txt",
            "document3.txt"
        ]

        selected_docs = []
        for doc in sample_docs:
            if st.checkbox(f"Use {doc}", value=True):
                selected_docs.append(doc)

        # Initialize system button
        if st.button("Initialize System", disabled=len(selected_docs) == 0 or not api_key):
            with st.spinner("Initializing RAG-QA System..."):
                try:
                    st.session_state.system = RAGQASystem(selected_docs)
                    st.session_state.initialized = True
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")

    # Main area for query input and results
    query = st.text_input("Ask a question:", placeholder="e.g., What are the key features of Product X?")

    if st.button("Submit", disabled=not st.session_state.initialized):
        if query:
            with st.spinner("Processing..."):
                # Check if system is initialized properly
                if st.session_state.system is not None:
                    result = st.session_state.system.query(query)

                    # Display the response
                    st.header("Answer")
                    st.write(result["response"])

                    # Display debug information
                    st.subheader("Debug Information")

                    # Processing details
                    st.write("**Processing Details:**")
                    st.write(f"Processing Time: {result.get('processing_time', 'N/A')}")
                    st.write(f"Agent Used: {result.get('agent_path', 'N/A')}")

                    if result.get('agent_path') == "rag_agent":
                        # Retrieved chunks
                        st.write("**Retrieved Chunks:**")

                        # Create tabs for each chunk instead of nested expanders
                        if result.get('retrieved_chunks', []):
                            chunk_tabs = st.tabs([f"Chunk {i+1} - {chunk.get('source', 'Unknown')}"
                                                for i, chunk in enumerate(result.get('retrieved_chunks', []))])

                            for i, tab in enumerate(chunk_tabs):
                                with tab:
                                    chunk = result.get('retrieved_chunks', [])[i]
                                    st.write(chunk.get('content', 'No content'))

                        # LLM Prompt
                        st.write("**LLM Prompt:**")
                        st.code(result.get('llm_prompt', 'No prompt available'), language="text")
                else:
                    st.error("System not properly initialized. Please initialize the system from the sidebar first.")
        else:
            st.warning("Please enter a question.")

    if not st.session_state.initialized:
        st.info("Please initialize the system from the sidebar first.")

    # Footer
    st.divider()
    st.caption("RAG-Powered Multi-Agent Q&A Assistant | Built with LangChain, LangGraph, and Groq")

if __name__ == "__main__":
    create_streamlit_app()