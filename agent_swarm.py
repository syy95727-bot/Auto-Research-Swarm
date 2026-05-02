import os
import json
import logging
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

# ==========================================
# 1. 配置与初始化 (Setup & Config)
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger("AutoResearchSwarm")

# 初始化 LLM 客户端
# 提示：如果你在国内使用，可以替换为国内大模型兼容 OpenAI 格式的 base_url 和 api_key
API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-real-api-key-here")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
# 推荐使用支持复杂推理的模型
MODEL_NAME = "gpt-4o-mini" 

# ==========================================
# 2. 定义数据结构 (Pydantic Models)
# ==========================================
class TaskPlan(BaseModel):
    thoughts: str = Field(description="思考为什么要这么拆解任务")
    steps: List[str] = Field(description="具体的执行步骤列表")

class DataAnalysisResult(BaseModel):
    is_successful: bool = Field(description="代码或分析是否执行成功")
    extracted_metrics: dict = Field(description="提取到的核心数据字典，如 {'YoY_Growth': '15%'}")
    error_log: str = Field(default="", description="如果失败，记录的报错信息")

# ==========================================
# 3. 核心 Agent 类 (Core Agents)
# ==========================================
class PlannerAgent:
    """项目经理：负责把一句话需求拆解为具体步骤"""
    def plan(self, user_query: str) -> TaskPlan:
        logger.info(f"Task received. Planner is decomposing: '{user_query}'")
        
        # 使用真实的 API 调用，并强制 LLM 返回 JSON 结构
        response = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个资深的金融数据项目经理。请将用户的宏大需求拆解为3个以内的数据抓取与分析步骤。"},
                {"role": "user", "content": user_query}
            ],
            response_format=TaskPlan,
        )
        plan_result = response.choices[0].message.parsed
        logger.info(f"Plan generated with {len(plan_result.steps)} steps.")
        return plan_result

class DataAnalystAgent:
    """数据极客：负责根据步骤提取数据（这里模拟提取，真实场景可接入数据库）"""
    def execute(self, step_description: str, context: str) -> DataAnalysisResult:
        logger.info(f"Data Analyst executing: {step_description}")
        
        response = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个数据分析师，请从提供的上下文中提取量化指标。如果没有找到，必须在 error_log 中说明。"},
                {"role": "user", "content": f"任务: {step_description}\n上下文数据: {context}"}
            ],
            response_format=DataAnalysisResult,
        )
        result = response.choices[0].message.parsed
        
        if not result.is_successful:
            logger.warning(f"Analysis failed or missing data: {result.error_log}")
            # 这里可以接入真实的 Reflection (反思) 和 Retry (重试) 逻辑
        else:
            logger.info(f"Metrics extracted successfully: {result.extracted_metrics}")
            
        return result

class WriterAgent:
    """总编：负责将所有数据汇总并撰写专业报告"""
    def write_report(self, original_query: str, raw_data: dict) -> str:
        logger.info("Writer is synthesizing final report...")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一位严肃的商业分析师。请根据提供的数据字典，撰写一段约200字的分析简报，拒绝任何捏造（幻觉）。"},
                {"role": "user", "content": f"原始需求: {original_query}\n可用数据: {json.dumps(raw_data, ensure_ascii=False)}"}
            ]
        )
        report = response.choices[0].message.content
        logger.info("Final report generated successfully.")
        return report

# ==========================================
# 4. 主工作流编排 (Workflow Orchestration)
# ==========================================
def run_swarm():
    logger.info("Initializing Auto-Research-Swarm...")
    
    query = "分析公司A的2025年度营收增长率及核心利润率"
    # 模拟 RAG 检索回来的生肉文本
    mock_rag_context = "2025年财报显示，公司A实现总营收1500万美元，同比增长22.5%。核心净利润率为14.2%，较去年有所下滑。"
    
    # 实例化 Agent
    planner = PlannerAgent()
    analyst = DataAnalystAgent()
    writer = WriterAgent()
    
    # 1. 规划
    plan = planner.plan(query)
    print(f"\n[Planner Thoughts]: {plan.thoughts}")
    
    # 2. 依次执行分析
    aggregated_metrics = {}
    for i, step in enumerate(plan.steps):
        res = analyst.execute(step, mock_rag_context)
        if res.is_successful:
            aggregated_metrics.update(res.extracted_metrics)
            
    # 3. 撰写报告
    final_markdown = writer.write_report(query, aggregated_metrics)
    
    print("\n" + "="*40)
    print("📈 FINAL RESEARCH REPORT")
    print("="*40)
    print(final_markdown)

if __name__ == "__main__":
    # 运行前请确保安装了依赖：pip install openai pydantic
    # export OPENAI_API_KEY="你的真实KEY"
    run_swarm()
