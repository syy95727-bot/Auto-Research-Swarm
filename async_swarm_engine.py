import os
import json
import asyncio
import logging
import time
from typing import List, Dict, Any
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, 
    format='\033[36m%(asctime)s\033[0m - [\033[33m%(levelname)s\033[0m] - \033[35m%(name)s\033[0m: %(message)s'
)
logger = logging.getLogger("AsyncMultiAgentSwarm")

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
MODEL = "gpt-4o"

class TokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def add(self, usage):
        if usage:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            
    def display_cost(self):
        logger.info(f"Session Consumed: Prompt={self.total_prompt_tokens}, Completion={self.total_completion_tokens}")

tracker = TokenTracker()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_sandbox",
            "description": "Execute Python scripts in an isolated sandbox for data aggregation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python script returning JSON."},
                    "dataset_id": {"type": "string", "description": "Local dataset ID."}
                },
                "required": ["code", "dataset_id"]
            }
        }
    }
]

async def execute_python_sandbox(code: str, dataset_id: str) -> str:
    logger.info(f"[Sandbox] Initializing container for {dataset_id}...")
    await asyncio.sleep(1.5)
    
    if "import pandas" not in code:
        return json.dumps({"status": "error", "message": "ModuleNotFoundError: No module named 'pandas'"})
    
    return json.dumps({"status": "success", "result": {"CAGR": "18.5%", "Market_Share": "32%"}})

async def async_data_agent(query: str, max_iterations: int = 3) -> str:
    messages = [
        {"role": "system", "content": "You are a senior data scientist agent. You can execute code in a sandbox. If errors occur, read logs and self-correct."},
        {"role": "user", "content": query}
    ]
    
    for iteration in range(max_iterations):
        logger.info(f"[Agent Cycle {iteration+1}/{max_iterations}] Reasoning...")
        
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tracker.add(response.usage)
        
        if response_message.tool_calls:
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.warning(f"[Tool] Executing {function_name}...")
                
                if function_name == "execute_python_sandbox":
                    tool_result = await execute_python_sandbox(
                        code=function_args.get("code"),
                        dataset_id=function_args.get("dataset_id")
                    )
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_result,
                    })
        else:
            logger.info("[Agent] Task completed.")
            return response_message.content
            
    return "Error: Reached maximum iterations."

async def main():
    logger.info("Starting Asynchronous Swarm Engine...")
    start_time = time.time()
    
    tasks = [
        "Analyze dataset_A01 and calculate 2024 CAGR",
        "Extract revenue data from dataset_B02 and run linear regression",
        "Compare market share between dataset_A01 and dataset_B02"
    ]
    
    results = await asyncio.gather(*(async_data_agent(task) for task in tasks))
    
    for i, res in enumerate(results):
        print(f"\n[Result {i+1}]:\n{res}")
        
    logger.info(f"Execution finished in {time.time() - start_time:.2f} seconds.")
    tracker.display_cost()

if __name__ == "__main__":
    asyncio.run(main())
