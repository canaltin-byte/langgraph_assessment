from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from tavily import TavilyClient
import os   
import spacy
from utils import Utils
import re
load_dotenv()

class UserInputValidator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.doc = None
        self.tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
        self.intent_file_path = os.path.join(os.path.dirname(__file__), "source", "intent_keywords.txt")
        self.llm = ChatOpenAI(model="gpt-4")
        
    def input_validation(self, text):
        if isinstance(text, dict):
            text = text.get('content', '')
        elif isinstance(text, list):
            text = text[-1].get('content') if isinstance(text[-1], dict) else text[-1]
        elif not isinstance(text, str):
            return "Error: Invalid input format"
        return text       
    
        
    def get_intent(self, text):
        try:
            intention_dict = Utils.read_txt_file(self.intent_file_path)
            if not intention_dict:
                return "Error: Could not read intent keywords file"
                
            if text is None:
                return "Error: Please process text first using get_company_name"
                
            location_intent_count = 0 
            investments_intent_count = 0
            business_model_intent_count = 0
            timeframe_intent_count = 0
            customers_intent_count = 0
            doc_text = self.nlp(text)
            for token in doc_text:
                if len(token.lemma_.lower()) > 2:
                    if token.lemma_.lower() in intention_dict.get("Location", []) or token.text.lower() in intention_dict.get("Location", []):
                        location_intent_count += 1
                    if token.lemma_.lower() in intention_dict.get("Business Model", []) or token.text.lower() in intention_dict.get("Business Model", []):
                        business_model_intent_count += 1
                    if token.lemma_.lower() in intention_dict.get("Investments", []) or token.text.lower() in intention_dict.get("Investments", []):
                        investments_intent_count += 1
                    if token.lemma_.lower() in intention_dict.get("Timeframe", []) or token.text.lower() in intention_dict.get("Timeframe", []): 
                        timeframe_intent_count += 1
                    if token.lemma_.lower() in intention_dict.get("Customers", []) or token.text.lower() in intention_dict.get("Customers", []):
                        customers_intent_count += 1
            
            counts = {
                "Location": location_intent_count,
                "Business Model": business_model_intent_count,
                "Investments": investments_intent_count,
                "Timeframe": timeframe_intent_count,
                "Customers": customers_intent_count
            }
            
            max_count = max(counts.values())
            max_intents = [intent for intent, count in counts.items() if count == max_count]
            if len(max_intents) == 1:
                return max_intents[0]
            else:
                return self.get_intent_from_llm(text)
        except Exception as e:
            return f"Error in intent analysis: {str(e)}"
    
    def get_company_name_from_llm(self, text, detail, company_name):
        if detail.strip() == "":
            prompt = f"""what is the name of the company in this sentence:
            Analyze following :{text}
            Return only the company name.
            """
            response = self.llm.invoke(prompt) 
            list_companies_with_same_name = self.list_companies_with_same_name(response.content)
            original_company_name = response.content
            return original_company_name, list_companies_with_same_name
        else:
            tavily_search_for_company_detail = self.tavily_search_for_multiple_companies_detail(company_name, detail)
            prompt = f"""
            Consider the industry of the company: 
            Company name :{company_name}
            Company detail :{detail}
            Company search result from tavily :{tavily_search_for_company_detail}
            Return the actual company name using all the information. Your answer should be only the real full name of the company.
            """            
            response = self.llm.invoke(prompt)   
            return company_name, [response.content]
    
    def tavily_search_for_multiple_companies_detail(self, company_name, detail):
        try:
            company_name = self.input_validation(company_name)
            response = self.tavily_client.search(
                query = f"Consider the industry of the company: {company_name} and {detail} Return only one full company name.",
                search_depth="advanced",
                include_answer=True,
                include_domains=[],
                max_results=5,
            )
            return response.get('answer', 'No answer found')
        except Exception as e:
            return {"error": f"Error searching for company: {str(e)}"}
        
    def list_companies_with_same_name(self, company_name):
        prompt = f"""List all the companies named {company_name}, if there is one and only one company return company name. 
        If there are more than one company with the same name, list each of the companies as Company Name, 
        Company Industry with comma. Do not write anything else.
        """
        response = self.llm.invoke(prompt)
        result = []
        for i in response.content.split("\n"):
            temp = re.sub(r'[\d.]', '', i)
            if temp != "":
                result.append(temp.strip())
        maximum_len = min(len(result), 5)
        return result[:maximum_len]
        
            
    def get_intent_from_llm(self, text):
        prompt = f"""Analyze the following text and select the SINGLE most relevant subject from these five options:            
        Text to analyze: "{text}"
        1. Location: Any geographical or place-related information (e.g., cities, countries, regions)
        2. Business Model: Any business structure, revenue model, or operational aspects (e.g., how a company operates)
        3. Investments: Any financial investments, funding, or monetary aspects (e.g., money, costs, funding)
        4. Timeframe: Any temporal information, deadlines, or time-related aspects (e.g., latest, recent, upcoming, dates, periods)
        5. Customers: Any customer-related information, target audience, or market segments (e.g., who buys or uses something)
        If the text is not related to any of these subjects, return "None"
        Return only one word: "Location", "Business Model", "Investments", "Timeframe", "Customers" or "None"

        """
        response = self.llm.invoke(prompt)
        return response.content
    
    def intention_clearity(self, text, intention_answer):
        if intention_answer == "None":
            return "ambiguous"
        elif intention_answer == "Location":
            prompt = f"""{text} - control for this sentence how many location type (e.g. HQ, stores, factories) is related. if only one location type is related return 'clear', if more than one location type are related return 'ambigious'. Multiple locations does not mean ambigous, only multiple location types are ambigous. Answer just in one word ambigous or clear
            """
        else:
            return "clear"
        response = self.llm.invoke(prompt)
        return response.content
    
            
