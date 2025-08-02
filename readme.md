# End-to-End Multi-AI Agent System with LangGraph, AstraDB, and Llama 3.1
<img width="681" alt="Screenshot 2024-10-14 at 9 49 07 PM" src="https://github.com/user-attachments/assets/ea3e99a5-54c6-457b-8fff-fd91dacc2351">


## Project Overview

This project implements a state-of-the-art Retrieval-Augmented Generation (RAG) system using LangGraph, AstraDB, and Llama 3.1.
It leverages a multi-agent architecture to efficiently coordinate complex tasks across agents, enabling advanced query understanding, contextual reasoning, and response generation.

A key innovation in this system is the integration of the HyQE (Hybrid Query Expansion and Reranking) framework, which enhances the quality of document retrieval by generating synthetic queries for candidate documents and reranking them based on relevance to the original user query. This significantly improves the precision and reliability of the LLM responses by ensuring only the most contextually appropriate documents are passed to the generation stage.

The project also features seamless integration of:

Vector storage and retrieval via AstraDB

LLM orchestration using LangGraph

Embedding models for similarity search and reranking

A modular and extensible multi-agent pipeline for robust task decomposition

Overall, this system demonstrates how modern AI infrastructure and RAG techniques can be combined with cutting-edge reranking logic to deliver highly accurate, context-aware answers at scale.

## Features

- **Multi-Agent Architecture**: Designed to handle a variety of complex tasks by leveraging multiple agents.
- **Efficient Vector Storage and Retrieval**: Uses AstraDB for scalable, serverless vector database management.
- **Integration of LangGraph**: Manages agent communication and state transitions.
- **Wikipedia Search Integration**: Allows dynamic external information retrieval beyond stored vectors.
- **Smart Routing Mechanism**: Decides between querying AstraDB or Wikipedia based on the task.
- **NLP Capabilities with Llama 3.1**: Utilizes the Llama model for natural language processing and routing decisions.

## Prerequisites

Before running the application, ensure the following dependencies and accounts are set up:

- **Python**: Version 3.7+
- **API Access**:
  - AstraDB account and application token
  - Hugging Face account and API token
  - Grok API key for Llama model access

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/multi-ai-agents-rag.git
   cd multi-ai-agents-rag
   ```

2. **Install required packages**:
   ```bash
   pip install langchain langraph astrapy tiktoken langchain-community chromadb huggingface_hub wikipedia matplotlib
   ```

3. **Set up environment variables**:
   - `ASTRA_DB_APPLICATION_TOKEN`: Your AstraDB application token
   - `ASTRA_DB_ID`: Your AstraDB database ID
   - `HF_TOKEN`: Your Hugging Face API token
   - `GROCK_API_KEY`: Your Grok API key

## Step-by-Step Implementation

### 1. Set up AstraDB

- Create a serverless vector database in AstraDB.
- Initialize the database connection using the provided application token and database ID.

### 2. Data Preparation

- Load documents from the web using `WebBaseLoader`.
- Analyze document lengths and visualize distributions for better understanding.
- Split the text into manageable chunks using `RecursiveCharacterTextSplitter` for optimized storage and retrieval.

### 3. Store Vectors in AstraDB

- Convert text into vectors using Hugging Face embeddings.
- Use `CassandraVectorStore` to insert the vectors into AstraDB.
- Visualize embedding dimensions and validate vector insertion.

### 4. Wikipedia Search Implementation

- Set up the Wikipedia API wrapper to enable external queries.
- Develop a function that fetches and processes information from Wikipedia.

### 5. Routing Mechanism

- Implement a `RouteQuery` class to intelligently decide whether to fetch data from AstraDB or query Wikipedia.
- Utilize Llama 3.1 for natural language processing tasks and decision-making.

### 6. Develop the LangGraph Workflow

- Define the workflow graph state and create nodes for Wikipedia search and vector retrieval.
- Add conditional edges based on task type to connect the nodes and compile the workflow.

### 7. Execute the Multi-Agent System

- Stream inputs through the workflow and capture outputs based on the selected route (AstraDB vector retrieval or Wikipedia).
- Measure execution times to analyze performance and optimize the workflow.

## Usage

To run the application:

```python
# Initialize the workflow
app = workflow.compile()

# Process a query
query = "What is an AI agent?"
for output in app.stream({"question": query}):
    print(f"{output['key']}: {output['value']}")
```

## Performance and Analysis

- **Time Tracking**: Measure the time taken for each stage (document loading, splitting, embedding, and vector insertion) to optimize processing times.
- **Visualization**: Display document length distributions and embedding dimensions to gain deeper insights into data characteristics.
- **Model Comparison**: Evaluate different embedding models to identify the most efficient option for the dataset and tasks.

## Future Improvements

- Add more external tools for diverse information retrieval.
- Enhance routing mechanisms with advanced NLP techniques for more precise decision-making.
- Scale vector storage and retrieval solutions for handling larger datasets efficiently.
- Develop a user-friendly interface for seamless interaction with the multi-agent system.

## Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request for review.


## Acknowledgments

- Appreciation for the LangChain and LangGraph communities for providing robust tools and documentation.
- Thanks to AstraDB for enabling efficient and scalable vector storage solutions.
- Gratitude to the Grok team for their Llama 3.1 model and its integration capabilities.

---

