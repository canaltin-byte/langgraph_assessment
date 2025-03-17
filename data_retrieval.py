import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class DataRetrieval:
    def __init__(self):
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o")
        
    def create_search_input(self, company_name, intent, text, refined_query):
        prompt = f"""Create JUST ONE search text for web search. 
        Company Name: "{company_name}"
        Intent: "{intent}"
        Text: "{text}"
        Take into consideration this refined query: "{refined_query}"
        """
        
        response = self.llm.invoke(prompt)
        return response.content

    def tavily_search(self, text):
        from tavily import TavilyClient
        client = TavilyClient(self.TAVILY_API_KEY)
        response = client.search(
            query=text,
            search_depth="advanced", 
            include_answer=True,     
            include_domains=[],       
            max_results=5           
        )
        url_sum = self.url_summary(response)
        return url_sum, response.get('answer')
    
    def url_summary(self, all_response):
        prompt=f"""{all_response} - summarize all the source names used in this text and do not include the URL itself"""
        url_sum = self.llm.invoke(prompt)
        return url_sum
    
    def search_wikipedia(self, query):
        """Searches Wikipedia and returns the summary of the first result."""
        from wikipedia import summary
        try:
            return summary(query, sentences=2)
        except:
            return "I couldn't find any information on that."

    def data_retrieval_general(self, text, intent):
        if intent.lower() == "customers":
            url_summary, data_retrieval_general = self.tavily_search(text)
            return url_summary, data_retrieval_general
        elif intent.lower() == "business model":
            url_summary, data_retrieval_general = self.tavily_search(text)
            return url_summary, data_retrieval_general
        elif intent.lower() == "timeframe":
            url_summary, data_retrieval_general = self.tavily_search(text)
            return url_summary, data_retrieval_general
        elif intent.lower() == "location":
            url_summary, data_retrieval_general_output = self.tavily_search(text)
            return url_summary, data_retrieval_general_output
        elif intent.lower() == "investments":
            url_summary, data_retrieval_general = self.tavily_search(text)
            return url_summary, data_retrieval_general