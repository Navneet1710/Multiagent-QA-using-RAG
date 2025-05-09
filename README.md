﻿# Multiagent-QA-using-RAG

A sophisticated question-answering system that combines Retrieval-Augmented Generation (RAG) with a multi-agent workflow to provide accurate, context-aware responses from your documents.

![RAG-Powered Multi-Agent Q&A Assistant](https://api.placeholder.com/800/300)

##  Overview

This project implements an intelligent Q&A system that:

- Uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on your documents
- Features a multi-agent architecture with specialized tools for different query types
- Automatically routes queries to the appropriate agent (RAG or tool-based)
- Provides detailed debug information about the retrieval and generation process
- Runs as a user-friendly Streamlit web application

##  Features

- **Document Processing**: Upload and process text documents for information retrieval
- **Intelligent Query Routing**: Automatically determines the best way to answer each question
- **Specialized Tools**: 
  - Calculator for mathematical expressions
  - Dictionary for word definitions
  - Unit conversion functionality
- **RAG Pipeline**: Retrieves relevant document chunks before generating answers
- **Debug Information**: View retrieved chunks, prompts used, and processing details
- **Modern UI**: Clean, responsive interface built with Streamlit

## 🛠️ Technologies Used

- [LangChain](https://www.langchain.com/) - Framework for LLM application development
- [LangGraph](https://github.com/langchain-ai/langgraph) - Flow control for LLM applications
- [ChromaDB](https://www.trychroma.com/) - Vector database for storing document embeddings
- [Sentence Transformers](https://www.sbert.net/) - Embedding model for semantic search
- [Groq](https://groq.com/) - Fast LLM inference API
- [Streamlit](https://streamlit.io/) - Web application framework
- [pysqlite3-binary](https://pypi.org/project/pysqlite3-binary/) - SQLite compatibility layer

##  Installation

### Prerequisites

- Python 3.9+
- Git
- A Groq API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Navneet1710/Multiagent-QA-using-RAG.git
   cd Multiagent-QA-using-RAG
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

##  Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. In the sidebar:
   - Select the sample documents you want to use
   - Click "Initialize System" to set up the RAG system

4. Enter your question in the text input field and click "Submit"

5. View the answer and debug information in the main panel

##  Contributing

Contributions are welcome! Here's how you can contribute to this project:

1. Fork the repository
   ```bash
   # Click the Fork button on GitHub or use:
   gh repo fork Navneet1710/Multiagent-QA-using-RAG
   ```

2. Create a new branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and commit them
   ```bash
   git add .
   git commit -m "Add your meaningful commit message"
   ```

4. Push to your branch
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request from your forked repository to the original one

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add comments for complex logic
- Update documentation when adding new features
- Write meaningful commit messages
- Test your changes before submitting a PR

##  Common Issues & Solutions

### SQLite Version Issues

If you encounter SQLite version compatibility problems:

```
RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
```

The package already includes a fix for this in `utils.py`:

```python
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

If you still encounter issues, make sure:
1. `pysqlite3-binary` is installed
2. The import code appears BEFORE importing chromadb

### Large File Issues with Git

If you have trouble pushing to GitHub due to large files:

1. Make sure your `.gitignore` file properly excludes the virtual environment:
   ```
   venv/
   ```

2. If you've already committed large files, remove them from git tracking:
   ```bash
   git rm -r --cached venv/
   git commit -m "Remove venv from git tracking"
   ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgements

- [LangChain](https://www.langchain.com/) for the LLM application framework
- [Groq](https://groq.com/) for the fast LLM inference API
- [Streamlit](https://streamlit.io/) for the web application framework
