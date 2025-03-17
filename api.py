from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from typing import Dict, Optional
from main import langgraph_entrapeer, State
from langgraph.types import Command, interrupt

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

conversations: Dict[str, dict] = {}

class Question(BaseModel):
    text: str

class Response(BaseModel):
    conversation_id: str
    message: str
    requires_input: bool
    final_answer: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

  

@app.post("/start_conversation")
async def start_conversation(question: Question):
    conversation_id = str(uuid.uuid4())
    thread = {"configurable": {"thread_id": conversation_id}}

    initial_input = {
        "input": question.text,
        "company_name": "",
        "company_detail": "",
        "company_list": [],
        "intent_detail": "",
        "search_input": "",
        "update_input": "",
        "needs_refinement": False,
        "refined_query": "",
        "evaluation_result": "",
        "data_retrieval_general_output": "",
        "intent": "",
        "intent_ambiguity": "",
        "feedback": "",
        "question_to_user": "",
        "data_retrieval_tavily_input": "",
        "data_retrieval_wikipedia_input": "",
        "data_retrieval_tavily_output": "",
        "data_retrieval_wikipedia_output": "",
        "dummy_state_input": "",
        "final_answer": "",
        "original_company_name": "",
        "url_summary": ""
    }

    conversations[conversation_id] = {
        "thread": thread,
        "waiting_for_input": False,
        "last_event": None
    }

    try:
        for event in langgraph_entrapeer.stream(initial_input, thread, stream_mode="updates"):
            
            if isinstance(event, dict):
                if "__interrupt__" in event:
                    conversations[conversation_id]["waiting_for_input"] = True
                    conversations[conversation_id]["last_event"] = event
                    interrupt_value = event["__interrupt__"][0]["value"] if isinstance(event["__interrupt__"], list) else event["__interrupt__"]
                    interrupt_value_str = str(interrupt_value)
                    # Find the start and end positions
                    value_start = interrupt_value_str.find("value=") + 7  # length of "value='"
                    value_end = interrupt_value_str.find(", resumable=")
                    # Extract the value
                    interrupt_value_sentence = interrupt_value_str[value_start:value_end-1]
                    # Add newlines to the message
                    formatted_message = interrupt_value_sentence.replace(". ", ".\n").replace("?", "?\n")
                    return Response(
                        conversation_id=conversation_id,
                        message=formatted_message,
                        requires_input=True
                    )

        state = langgraph_entrapeer.get_state(thread)
        final_answer = state.values.get("final_answer", "No answer available")
        return Response(
            conversation_id=conversation_id,
            message="Conversation complete",
            requires_input=False,
            final_answer=final_answer
        )

    except Exception as e:
        import traceback
        error_detail = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print("114", error_detail) 
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/continue_conversation/{conversation_id}")
async def continue_conversation(conversation_id: str, response: Question):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation = conversations[conversation_id]
    
    if not conversation["waiting_for_input"]:
        raise HTTPException(status_code=400, detail="No input expected for this conversation")
   
    try:
        for event in langgraph_entrapeer.stream(
            Command(resume=response.text),
            conversation["thread"],
            stream_mode="updates"
        ):
            if isinstance(event, dict) and "__interrupt__" in event:  
                conversation["last_event"] = event
                conversation["waiting_for_input"] = True  
                return Response(
                    conversation_id=conversation_id,
                    message=event["__interrupt__"][0]["value"],
                    requires_input=True
                )

        
        conversation["waiting_for_input"] = False
        final_answer = langgraph_entrapeer.get_state(conversation["thread"]).values.get("final_answer")
        conversations.pop(conversation_id) 

        return Response(
            conversation_id=conversation_id,
            message="Conversation complete",
            requires_input=False,
            final_answer=final_answer
        )

    except Exception as e:
        print("156", e) 
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/getResponse")
async def get_response(msg: str = Query(...)):
    conversation_id = str(uuid.uuid4())
    thread = {"configurable": {"thread_id": conversation_id}}

    initial_input = {
        "input": msg,
        "company_name": "",
        "company_detail": "",
        "company_list": [],
        "intent_detail": "",
        "search_input": "",
        "update_input": "",
        "needs_refinement": False,
        "refined_query": "",
        "evaluation_result": "",
        "data_retrieval_general_output": "",
        "intent": "",
        "intent_ambiguity": "",
        "feedback": "",
        "question_to_user": "",
        "data_retrieval_tavily_input": "",
        "data_retrieval_wikipedia_input": "",
        "data_retrieval_tavily_output": "",
        "data_retrieval_wikipedia_output": "",
        "dummy_state_input": "",
        "final_answer": "",
        "original_company_name": "",
        "url_summary": ""
    }

    try:
        for event in langgraph_entrapeer.stream(initial_input, thread, stream_mode="updates"):
            if type(event).__name__ == "Interrupt":                 
                return str(event.value) if hasattr(event, "value") else "Please provide more information"
            
        final_answer = langgraph_entrapeer.get_state(thread).values.get("final_answer")
        return final_answer if final_answer else "No answer available"

    except Exception as e:
        print("201", e) 
        raise HTTPException(status_code=500, detail=str(e))
