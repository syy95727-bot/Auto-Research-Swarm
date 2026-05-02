import os
import json
import asyncio
import logging
import time
from typing import List, Dict, Any
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# ==========================================
# 1. 高级异步日志与初始化
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='\033[36m%(asctime)s\033[0m - [\033[33m%(levelname)s\033[0m] - \033[35m%(name)s\033[0m: %(message)s'
)
logger = logging.getLogger("AsyncMultiAgentSwarm")

# 使用异步客户端 AsyncOpenAI，展现高并发处理能力
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-your-real-api-key"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
MODEL = "gpt-4o" # 使用高级模型

# 全局 Token 追踪器（申请额度时的最强力证据）
class TokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def add(self, usage):
        if usage:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            
    def display_cost(self):
        logger.info(f"📊 [Token Tracker] Session Consumed: Prompt={self.total_prompt_tokens}, Completion={self.total_completion_tokens}")

tracker = TokenTracker()

# ==========================================
# 2. 原生 Function Calling (工具定义)
# 告诉模型它可以调用外部 Python 环境
# ==========================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_in_wsl_sandbox",
            "description": "在隔离的沙盒中执行 Python (Pandas/Numpy) 代码来进行复杂的数据聚合与统计。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "需要执行的 Python 脚本，必须返回 JSON 序列化的结果"
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": "RAG 检索返回的本地数据集引用 ID"
                    }
                },
                "required": ["code", "dataset_id"]
            }
        }
    }
]

# 模拟真实的沙盒执行环境
async def execute_python_in_wsl_sandbox(code: str, dataset_id: str) -> str:
    logger.info(f"🚀 [Sandbox] Container spinning up for dataset {dataset_id}...")
    await asyncio.sleep(1.5)  # 模拟执行耗时
    
    # 模拟报错与重试逻辑 (注入不确定性)
    if "import pandas" not in code:
        return json.dumps({"status": "error", "message": "ModuleNotFoundError: No module named 'pandas'"})
    
    return json.dumps({"status": "success", "result": {"CAGR": "18.5%", "Market_Share": "32%"}})

# ==========================================
# 3. 异步 ReAct 核心循环 (Thought -> Action -> Observation)
# ==========================================
async def async_data_agent(query: str, max_iterations: int = 3) -> str:
    """带有自我纠错和工具调用的深度 ReAct Agent"""
    
    messages = [
        {"role": "system", "content": "你是一个高级数据科学家 Agent。你有权调用 Python 沙盒工具来计算数据。如果遇到错误，请阅读报错日志并修改代码重试。"},
        {"role": "user", "content": query}
    ]
    
    for iteration in range(max_iterations):
        logger.info(f"🧠 [Agent Cycle {iteration+1}/{max_iterations}] Thinking...")
        
        # 异步调用 API
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tracker.add(response.usage) # 记录 token 消耗
        
        # 判断模型是否决定调用工具 (Action)
        if response_message.tool_calls:
            messages.append(response_message) # 必须将 tool_calls 添加到历史
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.warning(f"🔧 [Tool Called] {function_name} with args:\n{function_args['code'][:50]}...")
                
                # 执行工具 (Observation)
                if function_name == "execute_python_in_wsl_sandbox":
                    tool_result = await execute_python_in_wsl_sandbox(
                        code=function_args.get("code"),
                        dataset_id=function_args.get("dataset_id")
                    )
                    
                    # 将执行结果返回给模型
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_result,
                    })
        else:
            # 模型认为任务完成，输出最终结论
            logger.info("✅ [Agent] Task completed.")
            return response_message.content
            
    return "Error: Reached maximum iterations without finalizing."

# ==========================================
# 4. 高并发多任务编排
# ==========================================
async def main():
    logger.info("Starting Asynchronous Swarm Execution Engine...")
    start_time = time.time()
    
    # 模拟同时并发处理三个复杂研报任务 (证明你为什么需要极高的并发限流额度, RPM/TPM)
    tasks = [
        "使用 pandas 分析 dataset_A01，计算 2024 年复合增长率",
        "提取 dataset_B02 中的营收数据并做线性回归预测",
        "对比前两者的市场占有率并输出结论"
    ]
    
    # asyncio.gather 实现并发执行
    results = await asyncio.gather(*(async_data_agent(task) for task in tasks))
    
    for i, res in enumerate(results):
        print(f"\n[Result {i+1}]:\n{res}")
        
    logger.info(f"All pipelines executed in {time.time() - start_time:.2f} seconds.")
    tracker.display_cost() # 打印最终烧掉的 Token 数量

if __name__ == "__main__":
    # 运行此脚本：python async_swarm_engine.py
    asyncio.run(main())
