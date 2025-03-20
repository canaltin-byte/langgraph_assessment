import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_retrieval import DataRetrieval

@pytest.fixture
def retrieval():
    """Create a UserInputValidator instance for testing."""
    with patch('tavily.TavilyClient') as mock_tavily, \
         patch('langchain_openai.ChatOpenAI') as mock_llm:
        # Mock the spaCy model
        mock_nlp = Mock()
        
        # Mock the Tavily client
        mock_tavily_instance = Mock()
        mock_tavily.return_value = mock_tavily_instance
        
        # Mock the LLM
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        retrieval = DataRetrieval()
        retrieval.tavily_client = mock_tavily_instance
        retrieval.llm = mock_llm_instance
        return retrieval


def test_create_search_input(retrieval):
    """Test the create_search_input method."""
    print("\n=== Testing create_search_input method ===")
    company_name = "Entrapeer"
    intent = "Location"
    text = "Where is the Entrapeer headquarters?"
    refined_query = "Find the Entrapeer main office location"
    
    print(f"Input parameters:")
    print(f"- Company: {company_name}")
    print(f"- Intent: {intent}")
    print(f"- Text: {text}")
    print(f"- Refined query: {refined_query}")
    
    result = retrieval.create_search_input(company_name, intent, text, refined_query)
    print(f"Result: {result}")    
    assert result is not None
    print("✓ Test passed successfully")

def test_tavily_search(retrieval):
    """Test the tavily_search method."""
    print("\n=== Testing tavily_search method ===")
    test_query = "Entrapeer headquarters location"
    print(f"Input query: {test_query}")
    
    url_sum, answer = retrieval.tavily_search(test_query)
    print(f"URL summary: {url_sum}")
    print(f"Answer: {answer}")
    
    assert len(answer) > 0
    print("✓ Test passed successfully")

def test_url_summary(retrieval):
    """Test the url_summary method."""
    print("\n=== Testing url_summary method ===")
    test_response = {
        'sources': ['Source 1', 'Source 2'],
        'answer': 'Test answer'
    }
    print(f"Input response: {test_response}")
    
    result = retrieval.url_summary(test_response)
    print(f"Result: {result}")
    assert result is not None
    print("✓ Test passed successfully")

def test_search_wikipedia(retrieval):
    """Test the search_wikipedia method."""
    print("\n=== Testing search_wikipedia method ===")
    test_query = "Entrapeer"
    print(f"Input query: {test_query}")
    
    result = retrieval.search_wikipedia(test_query)
    print(f"Result: {result}")
    
    assert len(result) > 0
    print("✓ Test passed successfully")

def test_data_retrieval_general_customers(retrieval):
    """Test data_retrieval_general method with customers intent."""
    print("\n=== Testing data_retrieval_general with customers intent ===")
    text = "Entrapeer customers"
    intent = "customers"
    print(f"Input text: {text}")
    print(f"Input intent: {intent}")
    
    url_sum, data = retrieval.data_retrieval_general(text, intent)
    print(f"URL summary: {url_sum}")
    print(f"Data: {data}")
    assert url_sum is not None
    assert data is not None
    print("✓ Test passed successfully")
