# LangGraph Company Information Assistant

A sophisticated natural language processing system built with LangGraph that helps users gather and analyze company information through a structured conversation flow.

## Features

- Company Name Resolution
  - Identifies company names from user queries
  - Handles disambiguation for companies with similar names
  - Uses Tavily search for company verification

- Intent Analysis
  - Identifies user intent across multiple categories:
    - Location information
    - Business model details
    - Investment information
    - Timeframe-related queries
    - Customer-related information
  - Uses spaCy for NLP processing
  - Handles intent ambiguity resolution

- Information Retrieval
  - Uses Tavily API for real-time web search
  - Structured data retrieval based on specific intents
  - Intelligent query refinement

## Technology Stack

- **Core Framework**: LangGraph for conversation flow management
- **Language Models**: 
  - OpenAI GPT-4 for natural language understanding
  - spaCy (en_core_web_trf) for NLP processing
- **APIs**:
  - Tavily API for web search
  - OpenAI API for language processing
- **Additional Libraries**:
  - langchain and langchain-core for LLM operations
  - FastAPI for API endpoints
  - pydantic for data validation

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_trf
   ```

3. Set up environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

4. Run the application:
   ```bash
   python main.py
   ```

## Project Structure

- `main.py`: Core application logic and LangGraph setup
- `text_analyze.py`: Text analysis and intent recognition
- `data_retrieval.py`: Information retrieval functions
- `evaluation.py`: Response evaluation and refinement
- `utils.py`: Utility functions
- `source/intent_keywords.txt`: Intent classification keywords
- `api.py`: FastAPI endpoints
- `server.py`: Server configuration

## Usage

The system accepts natural language queries about companies and provides structured responses based on the detected intent. Example queries:

- "Where is Tesla's headquarters located?"
- "Who are the customers of Entrapeer"
- "Tell me latest news about NVIDIA"
- "Which companies has Sequoia invested in?"


## Contributors

Can Altin

## Docker Setup

1. Make sure Docker and Docker Compose are installed on your system.

2. Build the Docker image:
```bash
docker-compose build
```

3. Start the application:
```bash
docker-compose up
```

4. To stop the application:
```bash
docker-compose down
```

- Rebuild the image (after dependency changes):
```bash
docker-compose build --no-cache
```

- Clean up Docker resources:
```bash
docker system prune -f
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

## API Endpoints

- `GET /`: Home page
- `POST /start_conversation`: Start a new conversation
- `GET /getResponse`: Get response from the AI

## Docker Configuration

The project includes the following Docker-related files:

- `Dockerfile`: Defines the container image
- `docker-compose.yml`: Orchestrates the application
- `.dockerignore`: Excludes unnecessary files from the build

### Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key

## License

This project is licensed under the MIT License - see the LICENSE file for details.