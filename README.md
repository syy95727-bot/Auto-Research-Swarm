# 🚀 Auto-Research-Swarm: 基于多智能体协作的自动化研报分析系统

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Supported-green)
![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-orange)
![Status](https://img.shields.io/badge/Status-Private_Deployment-red)

## 📌 项目简介

**Auto-Research-Swarm** 是一个面向复杂商业分析与投研场景的多智能体（Multi-Agent）协作框架。
传统单次 LLM 调用在面对超长文本和复杂跨步任务时，容易出现严重的“模型幻觉”与执行中断。本项目通过引入 ReAct 范式和具有自我纠错能力的智能体网络，将“研报生成”流程化、自动化，彻底打破分析与执行的壁垒。

## 🧠 核心架构图

本项目由 4 个核心 Agent 组成，采用有向无环图（DAG）的任务流转机制。
```mermaid
graph TD
    A[User Prompt 复杂需求] --> B(Task Planner Agent<br/>意图识别与任务拆解)
    B --> |生成执行计划| C(RAG & Researcher Agent<br/>信息检索与长文本浓缩)
    C --> |结构化上下文| D(Data Analyst Agent<br/>代码编写与数据执行)
    
    D --> E[(Python Sandbox<br/>沙盒执行环境)]
    E -- Error Log --> D
    D -. Self-Reflection / 自动纠错重写 .-> D
    
    E -- 执行成功 / 图表与数据 --> F(Reviewer Agent<br/>审查、交叉验证与撰写)
    F --> G([Final Report 最终研报])
    
    style B fill:#e1f5fe,stroke:#03a9f4
    style C fill:#e8f5e9,stroke:#4caf50
    style D fill:#fff3e0,stroke:#ff9800
    style F fill:#fce4ec,stroke:#e91e63
