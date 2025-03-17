import uuid
from dotenv import load_dotenv 
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from text_analyze import UserInputValidator
from data_retrieval import DataRetrieval
from evaluation import evaluate_and_refine

user_input_validator = UserInputValidator()
data_retrieval = DataRetrieval()
load_dotenv()

question_to_user = "Please ask your question:"

class State(TypedDict):
    input: str
    company_name: str
    company_list: list
    company_detail: str
    intent: str
    intent_ambiguity: str
    intent_detail: str
    feedback: str
    question_to_user: str    
    needs_refinement: bool
    refined_query: str
    evaluation_result: str 
    update_input: str
    search_input: str
    data_retrieval_general_output: str
    final_answer: str
    original_company_name: str
    url_summary: str

def extract_company_name(state: State) -> State:
    original_company_name, company_list = user_input_validator.get_company_name_from_llm(state["input"], state["company_detail"], state["company_name"])
    if len(company_list) == 1:
        return {"company_name": original_company_name, "company_list": company_list}
    else:
        return {"company_name": original_company_name, "company_list": company_list}

def listing_companies_with_same_name(state: State) -> State:
    return {"company_list": state["company_list"]}

def extract_intent(state: State) -> State:
    intent = user_input_validator.get_intent(state["input"] + " " + state["intent_detail"])
    return {"intent": intent}

def check_intent_ambiguity(state: State) -> State:
    intent = state["intent"]
    input = state["input"] + " " + state["intent_detail"]
    checking_intent_ambiguity = user_input_validator.intention_clearity(input, intent)
    return {"intent_ambiguity": checking_intent_ambiguity}

def anaysis_company_completed(state):
    if len(state['company_list']) == 1:
        return {"company_name": state['company_list'][0]}

def anaysis_question_completed(state):
    if len(state['company_list']) == 1 and "clear" in state['intent_ambiguity'].lower():
        create_search_input = data_retrieval.create_search_input(state['company_name'], state['intent'], state['input'] + " " + state['company_detail'] + " " + state['intent_detail'], state['refined_query'])
        
        return {"intent": state['intent'], 
                "update_input": state['input'] + " " + state['company_detail'] + " " + state['intent_detail'], 
                "search_input": create_search_input}

def data_retrieval_general(state):
    if not state["search_input"] or state["search_input"].isspace():
        return {"data_retrieval_general_output": "No valid search input provided"}
    url_summary, data_retrieval_general = data_retrieval.data_retrieval_general(state["search_input"], state["intent"])
    return {"data_retrieval_general_output": data_retrieval_general, "url_summary": url_summary}
    
def evaluate_and_refine_answer(state):
    needs_refinement, refined_query, evaluation_result = evaluate_and_refine(state["update_input"], state["data_retrieval_general_output"])
    return {"needs_refinement": needs_refinement, 
            "refined_query": refined_query, 
            "evaluation_result": evaluation_result}

def route_company_list(state):
    company_list = state["company_list"]
    company_string = ", ".join(company_list)
    if len(company_list) == 1:
        return "Accepted"
    print(f"I found multiple companies: {company_string} \n Please provide more details about the company you are looking for such as industry, function,and location")
    return "Rejected"
    
def route_intent_ambiguity(state):
    intent_ambiguity = state["intent_ambiguity"]
    if "clear" in intent_ambiguity.lower():
        return "Accepted"
    print(f"I found more than one location type. Could you please clarify what kind of location you are asking about, such as stores, headquarters, or factories?")
    return "Rejected"
    
def route_needs_refinement(state):
    needs_refinement = state["needs_refinement"]
    if needs_refinement:
        return "Rejected"
    else:
        return "Accepted"
    
    
def additional_question_for_company(state):
    company_list = state["company_list"]
    company_string = ", ".join(company_list)
    if state["company_detail"] == "":
        detail = interrupt(f"I found multiple companies: {company_string}. Please provide more details about the company you are looking for such as industry, function,and location")
    return {"company_detail": state["company_detail"] + " " + detail}

def additional_detail_for_intent(state):
    if state["intent"] == "Location":
        detail = interrupt("I found more than one location type. Could you please clarify what kind of location you are asking about, such as stores, headquarters, or factories?")
    else:
        detail = interrupt("I cannot be sure about your intention with that question can you be more specific about what you are looking for?")
    
    return {"intent_detail": state["intent_detail"] + " " + detail}

def final_answer_output(state):
    final_answer = state['data_retrieval_general_output']
    url_summary = state['url_summary'].content if state['url_summary'] else ""
    url_summary = url_summary.replace("\n", ", ")
    formatted_response = f"{final_answer}(Sources: {url_summary})"
    return {"final_answer": formatted_response}

builder = StateGraph(State)
builder.add_node("extract_company_name", extract_company_name)
builder.add_node("extract_intent", extract_intent)
builder.add_node("listing_companies_with_same_name", listing_companies_with_same_name)
builder.add_node("check_intent_ambiguity", check_intent_ambiguity)
builder.add_node("anaysis_question_completed", anaysis_question_completed)
builder.add_node("additional_question_for_company", additional_question_for_company)
builder.add_node("anaysis_company_completed", anaysis_company_completed)
builder.add_node("additional_detail_for_intent", additional_detail_for_intent)
builder.add_node("evaluate_and_refine_answer", evaluate_and_refine_answer)
builder.add_node("data_retrieval_general", data_retrieval_general)
builder.add_node("final_answer_output", final_answer_output)
builder.add_edge(START, "extract_company_name")
builder.add_edge("extract_company_name", "listing_companies_with_same_name")
builder.add_conditional_edges(
    "listing_companies_with_same_name",
    route_company_list,
    {  
        "Accepted": "anaysis_company_completed",
        "Rejected": "additional_question_for_company",
    },
)
builder.add_edge("additional_question_for_company", "extract_company_name")
builder.add_edge("anaysis_company_completed", "extract_intent")
builder.add_edge("extract_intent", "check_intent_ambiguity")
builder.add_conditional_edges(
    "check_intent_ambiguity",
    route_intent_ambiguity,
    {  
        "Accepted": "anaysis_question_completed",
        "Rejected": "additional_detail_for_intent",
    },
)
builder.add_edge("additional_detail_for_intent", "extract_intent")
builder.add_edge("anaysis_question_completed", "data_retrieval_general")
builder.add_edge("data_retrieval_general", "evaluate_and_refine_answer")
builder.add_conditional_edges(
    "evaluate_and_refine_answer",
    route_needs_refinement,
    {
        "Accepted": "final_answer_output",
        "Rejected": "anaysis_question_completed",
    },
)
builder.add_edge("final_answer_output", END)
memory = MemorySaver()

langgraph_entrapeer = builder.compile(checkpointer=memory)


