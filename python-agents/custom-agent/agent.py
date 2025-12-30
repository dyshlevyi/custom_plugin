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
    