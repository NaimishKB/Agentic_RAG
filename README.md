# Agentic_RAG

A complete RAG (Retrieval-Augmented Generation) pipeline built with LangChain, utilizing a hybrid vector store and a multi-agent system for enhanced information retrieval and question-answering. This project demonstrates advanced techniques in RAG, including re-ranking, diverse result expansion, and graph-based knowledge extraction.

## 🏗 Architecture

The system is composed of three main layers:

1.  **Ingestion Layer**:
    -   **Corpus**: Parses and processes raw text documents.
    -   **Text Splitter**: Splits large documents into manageable chunks using a recursive character text splitter.
    -   **Embedding**: Generates embeddings using the `SentenceTransformerEmbedder` and stores them in a FAISS index.
    -   **Graph Extraction**: Extracts relationships and entities to build a Neo4j knowledge graph.

2.  **Agentic Retrieval Layer**:
    -   **Hybrid Vector Store**: Combines FAISS (for speed) and ChromaDB (for persistence) indices.
    -   **Re-Ranker**: Uses `CohereReRanker` to re-rank the initial retrieval results for better relevance.
    -   **Expansion Agent**: A specialized agent that expands the initial query to retrieve more diverse and comprehensive results.

3.  **Agentic Generation Layer**:
    -   **Multi-Agent System**: A team of agents collaborate to answer complex queries.
    -   **Supervisor**: Orchestrates the workflow and aggregates the final answer.

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   `lfs` (Git Large File Storage) for handling large model files.

### Installation

1.  **Clone the repository**:
    ```bash
    git lfs clone https://github.com/NaimishKB/Agentic_RAG.git
    cd Agentic_RAG
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Create a `.env` file in the root directory with the following variables:

```env
GEMINI_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
HF_TOKEN=your_hf_token
Groq_key=your_groq_key
LANGSMITH_API_KEY=your_langsmith_key
SERP_API_KEY=your_serp_key
PERPLEXITY_KEY=your_perplexity_key
PINECONE_API_KEY=your_pinecone_key
INDEX_NAME=your_pinecone_index_name
NEO4J_CLIENT_ID=your_neo4j_client_id
NEO4J_CLIENT_SECRET=your_neo4j_client_secret
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
AURA_INSTANCEID=your_aura_instance_id
AURA_INSTANCENAME=your_aura_instance_name
```

> **Note**: For large file handling (models, indices), ensure Git LFS is installed and configured.