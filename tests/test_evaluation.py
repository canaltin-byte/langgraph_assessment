import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import AnswerEvaluator, evaluate_and_refine

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    with patch('langchain_openai.ChatOpenAI') as mock_llm:
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        yield mock_llm_instance

@pytest.fixture
def evaluator(mock_llm):
    """Create an AnswerEvaluator instance for testing."""
    return AnswerEvaluator(mock_llm)

def test_evaluate_answer_good_response(evaluator, mock_llm):
    """Test evaluate_answer with a good response."""
    print("\n=== Testing evaluate_answer with good response ===")
    query = "What is Entrapeer's business model?"
    answer = "Entrapeer is a technology company focused on AI solutions."
    
    mock_response = """Relevance Score: 8
Completeness Score: 7
Missing Information:
Refinement Needed: No
Refined Query: ''"""
    
    mock_llm.invoke.return_value = Mock(content=mock_response)
    
    print(f"Input query: {query}")
    print(f"Input answer: {answer}")
    
    result = evaluator.evaluate_answer(query, answer)
    print(f"Evaluation result: {result}")
    
    assert result['relevance_score'] >= 6
    assert result['completeness_score'] >= 6
    assert result['missing_information'] == []
    assert result['refinement_needed'] is False
    print("✓ Test passed successfully")

def test_evaluate_answer_needs_refinement(evaluator, mock_llm):
    """Test evaluate_answer with a response that needs refinement."""
    print("\n=== Testing evaluate_answer with response needing refinement ===")
    query = "What is Entrapeer's business model?"
    answer = "Entrapeer is a company."
    
    mock_response = """Relevance Score: 4
Completeness Score: 3
Missing Information: business model details, revenue streams, target market
Refinement Needed: Yes
Refined Query: What are Entrapeer's business model, revenue streams, and target market?"""
    
    mock_llm.invoke.return_value = Mock(content=mock_response)
    
    print(f"Input query: {query}")
    print(f"Input answer: {answer}")
    
    result = evaluator.evaluate_answer(query, answer)
    print(f"Evaluation result: {result}")
    
    assert result['relevance_score'] < 5
    assert result['completeness_score'] < 5
    assert len(result['missing_information']) >= 1
    assert result['refinement_needed'] is True
    assert result['refined_query'] is not None
    print("✓ Test passed successfully")

def test_needs_refinement_above_threshold(evaluator):
    """Test needs_refinement when scores are above threshold."""
    print("\n=== Testing needs_refinement with scores above threshold ===")
    evaluation_result = {
        'relevance_score': 8,
        'completeness_score': 7,
        'missing_information': [],
        'refinement_needed': False,
        'refined_query': "None"
    }
    
    print(f"Input evaluation result: {evaluation_result}")
    
    needs_refinement, refined_query = evaluator.needs_refinement(evaluation_result)
    print(f"Needs refinement: {needs_refinement}")
    print(f"Refined query: {refined_query}")
    
    assert needs_refinement is False
    assert refined_query == 'None'
    print("✓ Test passed successfully")

def test_needs_refinement_below_threshold(evaluator):
    """Test needs_refinement when scores are below threshold."""
    print("\n=== Testing needs_refinement with scores below threshold ===")
    evaluation_result = {
        'relevance_score': 3,
        'completeness_score': 4,
        'missing_information': ['details'],
        'refinement_needed': True,
        'refined_query': 'Improved query'
    }
    
    print(f"Input evaluation result: {evaluation_result}")
    
    needs_refinement, refined_query = evaluator.needs_refinement(evaluation_result)
    print(f"Needs refinement: {needs_refinement}")
    print(f"Refined query: {refined_query}")
    
    assert needs_refinement is True
    assert refined_query == 'Improved query'
    print("✓ Test passed successfully")

def test_evaluate_and_refine(mock_llm):
    """Test the evaluate_and_refine function."""
    print("\n=== Testing evaluate_and_refine function ===")
    query = "What is Entrapeer's business model?"
    answer = "Entrapeer is a technology company."
    
    mock_response = """Relevance Score: 6
Completeness Score: 5
Missing Information: business model details
Refinement Needed: Yes
Refined Query: What are Entrapeer's business model and revenue streams?"""
    
    mock_llm.invoke.return_value = Mock(content=mock_response)
    
    print(f"Input query: {query}")
    print(f"Input answer: {answer}")
    
    needs_refinement, refined_query, evaluation_result = evaluate_and_refine(query, answer)
    print(f"Needs refinement: {needs_refinement}")
    print(f"Refined query: {refined_query}")
    print(f"Evaluation result: {evaluation_result}")
    
    assert needs_refinement is True
    assert refined_query is not None
    assert evaluation_result['relevance_score'] <= 6
    assert evaluation_result['completeness_score'] <= 5
    print("✓ Test passed successfully") 