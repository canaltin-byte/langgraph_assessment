import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_analyze import UserInputValidator

@pytest.fixture
def validator():
    """Create a UserInputValidator instance for testing."""
    with patch('spacy.load') as mock_spacy_load, \
         patch('tavily.TavilyClient') as mock_tavily, \
         patch('langchain_openai.ChatOpenAI') as mock_llm:
        # Mock the spaCy model
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        # Mock the Tavily client
        mock_tavily_instance = Mock()
        mock_tavily.return_value = mock_tavily_instance
        
        # Mock the LLM
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        validator = UserInputValidator()
        validator.nlp = mock_nlp
        validator.tavily_client = mock_tavily_instance
        validator.llm = mock_llm_instance
        return validator

def test_input_validation_string(validator):
    """Test input validation with string input."""
    test_input = "test input"
    result = validator.input_validation(test_input)
    assert result == test_input

def test_input_validation_dict(validator):
    """Test input validation with dictionary input."""
    test_input = {"content": "test input"}
    expected = "test input"
    result = validator.input_validation(test_input)
    assert result == expected

def test_input_validation_list(validator):
    """Test input validation with list input."""
    test_input = [{"content": "test input"}]
    expected = "test input"
    result = validator.input_validation(test_input)
    assert result == expected

def test_input_validation_invalid(validator):
    """Test input validation with invalid input."""
    test_input = 123
    expected = "Error: Invalid input format"
    result = validator.input_validation(test_input)
    assert result == expected

def test_get_intent_success(validator):
    """Test successful intent detection."""
    # Mock the token behavior
    mock_token = Mock()
    mock_token.lemma_.lower.return_value = "location"
    mock_token.text.lower.return_value = "location"
    validator.nlp.return_value = [mock_token]
    
    # Mock the intent keywords file
    with patch('utils.Utils.read_txt_file') as mock_read_file:
        mock_read_file.return_value = {
            "Location": ["location"],
            "Business Model": [],
            "Investments": [],
            "Timeframe": [],
            "Customers": []
        }
        
        result = validator.get_intent("Where is the location?")
        assert result == "Location"

def test_get_intent_empty(validator):
    """Test intent detection with empty input."""
    result = validator.get_intent(None)
    assert result == "Error: Please process text first using get_company_name"

def test_get_intent_file_error(validator):
    """Test intent detection when file can't be read."""
    with patch('utils.Utils.read_txt_file') as mock_read_file:
        mock_read_file.return_value = None
        result = validator.get_intent("test input")
        assert result == "Error: Could not read intent keywords file"

def test_get_company_name_from_llm_no_detail(validator):
    """Test company name extraction without detail."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = "Entrapeer"
    validator.llm.invoke.return_value = mock_response
    
    # Mock list_companies_with_same_name
    validator.list_companies_with_same_name = Mock(return_value=["Entrapeer"])
    
    result = validator.get_company_name_from_llm("Tell me about Entrapeer", "", "")
    assert result == ("Entrapeer", ["Entrapeer"])

def test_get_company_name_from_llm_with_detail(validator):
    """Test company name extraction with detail."""
    # Mock Tavily response
    validator.tavily_search_for_multiple_companies_detail = Mock(return_value="Entrapeer Detail")
    
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = "Entrapeer"
    validator.llm.invoke.return_value = mock_response
    
    result = validator.get_company_name_from_llm("Tell me about Entrapeer", "technology", "Entrapeer")
    assert result == ("Entrapeer", ["Entrapeer"])

def test_tavily_search_for_multiple_companies_detail_error(validator):
    """Test Tavily search with error."""
    validator.tavily_client.search.side_effect = Exception("API Error")
    
    result = validator.tavily_search_for_multiple_companies_detail("Entrapeer", "technology")
    assert "Error searching for company" in result["error"]

def test_list_companies_with_same_name(validator):
    """Test listing companies with same name."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = "Entrapeer, Technology\nEntrapeer, Software"
    validator.llm.invoke.return_value = mock_response
    
    result = validator.list_companies_with_same_name("Entrapeer")
    assert isinstance(result, list)
    assert len(result) > 1
    assert all(isinstance(name, str) for name in result)

def test_get_intent_from_llm(validator):
    """Test LLM-based intent detection."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = "Location"
    validator.llm.invoke.return_value = mock_response
    
    result = validator.get_intent_from_llm("Where is the location of Entrapeer?")
    assert result == "Location"

def test_intention_clearity_location_clear(validator):
    """Test intention clarity with Location intention."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = "clear"
    validator.llm.invoke.return_value = mock_response
    
    result = validator.intention_clearity("Where is the headquarters?", "Location")
    assert result == "clear"

def test_intention_clearity_location_ambiguous(validator):
    """Test intention clarity with ambiguous Location intention."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = "ambiguous"
    validator.llm.invoke.return_value = mock_response
    
    result = validator.intention_clearity("Where are the Entrapeer", "Location")
    assert result == "ambiguous"

def test_intention_clearity_other(validator):
    """Test intention clarity with other intentions."""
    result = validator.intention_clearity("What is the business model of Entrapeer?", "Business Model")
    assert result == "clear"


