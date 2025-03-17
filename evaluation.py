from typing import Dict, Tuple, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

class AnswerEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.evaluation_prompt = """You are an answer quality evaluator. Analyze the given answer based on two main criteria:

1. Relevance (0-10):
   - Does the information directly address the user's question?
   - Is the information specific to the query?
   - Are there any irrelevant details?

2. Completeness (0-10):
   - Does the answer provide sufficient detail?
   - Are there any missing key aspects?
   - Is the information comprehensive enough?

User Query: {query}
Answer to Evaluate: {answer}

Provide your evaluation in this format:
Relevance Score: [0-10]
Completeness Score: [0-10]
Missing Information: [List any missing key aspects]
Refinement Needed: [Yes/No]
Refined Query: [If refinement is needed, provide an improved query]"""

    def evaluate_answer(self, query: str, answer: str) -> Dict:
        evaluation_message = self.evaluation_prompt.format(
            query=query,
            answer=answer
        )
        response = self.llm.invoke([
            SystemMessage(content="You are an answer quality evaluator."),
            HumanMessage(content=evaluation_message)
        ])
        evaluation_text = response.content
        try:
            lines = evaluation_text.split('\n')
            result = {
                'relevance_score': 0,
                'completeness_score': 0,
                'missing_information': [],
                'refinement_needed': False,
                'refined_query': None
            }
            for line in lines:
                line = line.strip()
                if line.startswith('Relevance Score:'):
                    result['relevance_score'] = int(line.split(':')[1].strip())
                elif line.startswith('Completeness Score:'):
                    result['completeness_score'] = int(line.split(':')[1].strip())
                elif line.startswith('Missing Information:'):
                    missing = line.split(':')[1].strip()
                    if missing and missing != 'None':
                        result['missing_information'] = [item.strip() for item in missing.split(',')]
                elif line.startswith('Refinement Needed:'):
                    result['refinement_needed'] = line.split(':')[1].strip().lower() == 'yes'
                elif line.startswith('Refined Query:'):
                    result['refined_query'] = line.split(':')[1].strip()
            
            return result
            
        except Exception as e:
            print(f"Error parsing evaluation response: {str(e)}")
            return {
                'relevance_score': 0,
                'completeness_score': 0,
                'missing_information': ['Error in evaluation'],
                'refinement_needed': True,
                'refined_query': query
            }

    def needs_refinement(self, evaluation_result: Dict) -> Tuple[bool, str]:
        threshold = 5
        
        needs_refinement = (
            evaluation_result['relevance_score'] < threshold or
            evaluation_result['completeness_score'] < threshold
        )
        
        refined_query = evaluation_result.get('refined_query', '')
        
        return needs_refinement, refined_query

def evaluate_and_refine(query: str, answer: str) -> Tuple[bool, str, Dict]:
    """
    Evaluate an answer and determine if it needs refinement.
    
    Args:
        query: The original user query
        answer: The answer to evaluate
        llm: The language model to use for evaluation
        
    Returns:
        Tuple of (needs_refinement: bool, refined_query: str, evaluation_result: Dict)
    """
    llm = ChatOpenAI(model="gpt-4o")
    evaluator = AnswerEvaluator(llm)
    evaluation_result = evaluator.evaluate_answer(query, answer)
    needs_refinement, refined_query = evaluator.needs_refinement(evaluation_result)
    
    return needs_refinement, refined_query, evaluation_result
