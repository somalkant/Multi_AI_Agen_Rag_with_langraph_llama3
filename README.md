# Enhanced Multi-AI Agent RAG System with MCP Agent Integration

This project extends the original HyQE RAG system with an MCP (Model Context Protocol) agent to provide enhanced capabilities for real-time information retrieval and tool usage.

## Overview

The enhanced system now includes three main components:

1. **Vector Store Agent** - For queries related to AI agents, prompt engineering, and adversarial attacks
2. **Wikipedia Search Agent** - For general knowledge questions and historical facts
3. **MCP Agent** - For real-time information, current events, web searches, and tool interactions

## Key Features

### Enhanced Routing Logic
The system now uses an intelligent router that determines the most appropriate agent based on the query type:

- **Vector Store**: Specialized knowledge about AI agents, prompt engineering, and adversarial attacks
- **Wikipedia**: General knowledge, historical facts, and biographical information
- **MCP Agent**: Real-time data, current events, travel queries, and web search capabilities

### MCP Agent Capabilities
The MCP agent can utilize various tools including:
- Web search (DuckDuckGo, Google)
- Airbnb accommodation search
- Browser automation (Playwright)
- Real-time data retrieval

### HyQE Reranking
All agents use HyQE (Hypothetical Query Embedding) reranking for improved document relevance.

## Files Structure

```
├── hyqe_rag3.ipynb                    # Original notebook
├── hyqe_rag3_with_mcp.ipynb          # Enhanced notebook with MCP agent
├── enhanced_rag_with_mcp.py          # Python script version
├── mcp_use.py                        # MCP agent implementation
├── browser_mcp.json                  # MCP server configuration
├── app.py                           # Original MCP chat application
└── README_enhanced.md               # This file
```

## Installation

1. Install required packages:
```bash
pip install langchain langgraph cassio langchain_community tiktoken langchain-groq langchainhub chromadb langchain_huggingface sentence_transformers arxiv wikipedia python-dotenv
```

2. Set up environment variables in `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

3. Ensure you have the AstraDB credentials configured in the script.

## Usage

### Running the Enhanced Notebook
1. Open `hyqe_rag3_with_mcp.ipynb` in Jupyter
2. Run all cells to initialize the system
3. Use the interactive testing function to query the system

### Running the Python Script
```bash
python enhanced_rag_with_mcp.py
```

### Interactive Mode
Uncomment the `interactive_test()` function call to run in interactive mode.

## Query Examples

### Vector Store Queries
- "What is agent?"
- "What are the types of agent memory?"
- "How does prompt engineering work?"

### Wikipedia Queries
- "Who is Barack Obama?"
- "What is the history of artificial intelligence?"
- "Tell me about quantum computing"

### MCP Agent Queries
- "What's the current weather in New York?"
- "Find me Airbnb accommodations in Paris"
- "What are the latest news about AI?"
- "Search for information about recent developments in machine learning"

## System Architecture

```
User Query
    ↓
Enhanced Router (LLM-based)
    ↓
┌─────────────┬─────────────┬─────────────┐
│ Vector Store│  Wikipedia  │  MCP Agent  │
│   Agent     │   Agent     │   Agent     │
└─────────────┴─────────────┴─────────────┘
    ↓             ↓             ↓
HyQE Reranking  HyQE Reranking  Tool Execution
    ↓             ↓             ↓
Document Results  Wiki Results   MCP Response
```

## Routing Logic

The enhanced router uses the following guidelines:

1. **Vector Store**: Questions about AI agents, prompt engineering, or adversarial attacks
2. **Wikipedia**: General knowledge, historical facts, or biographical information  
3. **MCP Agent**: Current events, real-time data, travel queries, or when live web search is needed

## MCP Configuration

The `browser_mcp.json` file configures the following MCP servers:
- **Playwright**: Browser automation
- **Airbnb**: Accommodation search
- **DuckDuckGo**: Web search
- **Google Search**: Alternative web search

## Customization

### Adding New MCP Tools
1. Update `browser_mcp.json` with new server configurations
2. Modify the `_determine_tools` method in `MCPAgent` class
3. Add corresponding tool execution logic

### Modifying Routing Logic
1. Update the routing prompt in the system variable
2. Adjust the `RouteQuery` model if needed
3. Modify the `route_question` function for custom routing logic

### Adding New Agents
1. Create a new agent function following the existing pattern
2. Add it to the graph in `build_enhanced_graph()`
3. Update the routing logic to include the new agent

## Performance Considerations

- The system uses HyQE reranking which may increase response time
- MCP agent calls are asynchronous to prevent blocking
- Vector store queries are optimized with similarity scoring
- Wikipedia queries are limited to top results for performance

## Error Handling

The system includes comprehensive error handling:
- Graceful fallbacks for failed tool executions
- Context length management for long conversations
- Retry mechanisms for transient failures
- Default routing to MCP agent for unknown query types

## Future Enhancements

Potential improvements include:
- Caching mechanisms for frequently accessed data
- Parallel execution of multiple agents
- Advanced query understanding and intent classification
- Integration with more MCP servers and tools
- Real-time learning from user feedback

## Troubleshooting

### Common Issues

1. **MCP Agent Errors**: Ensure MCP servers are properly configured and running
2. **Vector Store Issues**: Check AstraDB connection and credentials
3. **LLM Errors**: Verify Groq API key and rate limits
4. **Import Errors**: Install all required dependencies

### Debug Mode
Enable debug logging by setting environment variables or modifying the code to include more verbose output.

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is based on the original HyQE RAG system and extends it with MCP agent capabilities. 