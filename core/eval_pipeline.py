import os
import asyncio
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIConnectionError

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LLMOps.Evaluator")

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
JUDGE_MODEL = "gpt-4o"

class AgentEvaluationResult(BaseModel):
    chain_of_thought: str = Field(description="裁判模型对目标输出的深度分析与推理过程")
    hallucination_score: float = Field(description="幻觉指数，0.0 为完全基于事实，1.0 为严重捏造", ge=0.0, le=1.0)
    instruction_following_score: int = Field(description="指令遵循度评分，1-10分", ge=1, le=10)
    critical_errors: List[str] = Field(description="发现的任何致命逻辑错误或数据错误列表")
    is_usable_for_finetuning: bool = Field(description="该条数据是否足够优秀，可直接纳入后续微调 (SFT) 数据集")

class LLMAsAJudgePipeline:
    def __init__(self, concurrency_limit: int = 50):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.success_count = 0
        self.fail_count = 0

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1.5, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    async def evaluate_single_trace(self, trace_id: str, source_context: str, agent_output: str) -> AgentEvaluationResult:
        async with self.semaphore:
            logger.debug(f"Starting evaluation for Trace ID: {trace_id}")
            
            prompt = f"""
            你是一个严苛的数据质检专家。请评估以下 Agent 的输出。
            [原始上下文]: {source_context}
            [Agent 输出]: {agent_output}
            """
            
            response = await client.beta.chat.completions.parse(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "你必须进行极其严格的事实核查。"},
                    {"role": "user", "content": prompt}
                ],
                response_format=AgentEvaluationResult,
                temperature=0.1
            )
            
            return response.choices[0].message.parsed

    async def run_batch_evaluations(self, dataset: List[Dict[str, Any]]):
        logger.info(f"🚀 Initializing Batch Evaluation Pipeline for {len(dataset)} traces...")
        logger.info(f"⚙️ Concurrency Limit Set To: {self.semaphore._value}")
        
        tasks = [
            self.evaluate_single_trace(
                trace_id=data["id"], 
                source_context=data["context"], 
                agent_output=data["output"]
            )
            for data in dataset
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        finetune_dataset = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"❌ Trace {dataset[i]['id']} failed completely after retries: {str(res)}")
                self.fail_count += 1
            else:
                self.success_count += 1
                if res.is_usable_for_finetuning:
                    finetune_dataset.append(dataset[i])
                    
        logger.info(f"📊 Pipeline Completed. Success: {self.success_count}, Failed: {self.fail_count}")
        logger.info(f"💎 Yielded {len(finetune_dataset)} high-quality samples for Fine-Tuning.")
        return finetune_dataset

if __name__ == "__main__":
    mock_production_logs = [
        {"id": f"TRACE-{i:04d}", "context": "2025Q4 净利润 400万...", "output": "报告显示利润为 400万。"}
        for i in range(1, 105)
    ]
    
    pipeline = LLMAsAJudgePipeline(concurrency_limit=20)
    asyncio.run(pipeline.run_batch_evaluations(mock_production_logs))
