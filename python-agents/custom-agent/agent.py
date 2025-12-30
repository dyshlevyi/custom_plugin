import dataiku
from dataiku.llm.python import BaseLLM

# Add code here to define tools for instance

class MyLLM(BaseLLM):
    def __init__(self):
        pass

    def process(self, query, settings, trace):
        prompt = query["messages"][-1]["content"]

        # Add code here, for LLM completion, tool execution, etc...

        return {"text": "the agent answer"}
    
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain.tools import tool

from dataiku.llm.python import BaseLLM
from dataiku.langchain.dku_llm import DKUChatModel
from dataiku.langchain import LangchainToDKUTracer

import dataiku
from dataiku import SQLExecutor2
from duckduckgo_search import DDGS
from dataiku.sql import Constant, toSQL, Dialects

OPENAI_CONNECTION_NAME = "REPLACE_WITH_YOUR_CONNECTION_NAME"
model = DKUChatModel(llm_id=f"openai:{OPENAI_CONNECTION_NAME}:gpt-4o-mini")

@tool
def search_company_info(company_name: str) -> str:
    """
    Use this tool when you need to retrieve information on a company.
    The input of this tool is the company name.
    The output is either a small recap of the company or "No information …"
    meaning that we couldn't find information # about this company
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(f"{company_name} (company)", max_results=1))
        if results:
            return f"Information found about {company_name}: {results[0]['body']}"
        return f"No information found about {company_name}"
    

def generate_get_customer(dataset_name: str):
    @tool
    def get_customer(customer_id: str) -> str:
        """Get customer name, position and company information from database.
        The input is a customer id (stored as a string).
        The ouput is a string of the form:
            "The customer's name is \"{name}\", holding the position \"{job}\" at the company named {company}"
        """
        dataset = dataiku.Dataset(dataset_name)
        table_name = dataset.get_location_info().get('info', {}).get('quotedResolvedTableName')
        executor = SQLExecutor2(dataset=dataset)
        cid = Constant(str(customer_id))
        escaped_cid = toSQL(cid, dialect=Dialects.POSTGRES)  # Replace by your DB
        query_reader = executor.query_to_iter(
            f"""SELECT "name", "job", "company" FROM {table_name} WHERE "id" = {escaped_cid}""")
        for (name, job, company) in query_reader.iter_tuples():
            return f"The customer's name is \"{name}\", holding the position \"{job}\" at the company named {company}"
        return f"No information can be found about the customer {customer_id}"
    return get_customer