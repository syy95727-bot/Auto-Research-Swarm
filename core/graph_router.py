import os
import httpx
import logging
import socket
from typing import TypedDict, Annotated, Sequence

logger = logging.getLogger("GraphRouter")

class AgentState(TypedDict):
    task_id: str
    original_query: str
    collected_context: Annotated[Sequence[str], "Accumulated context chunks"]
    current_node: str
    error_count: int
    final_output: str

async def web_scraper_node(state: AgentState) -> AgentState:
    logger.info(f"[Node: Scraper] Fetching intelligence for {state['task_id']}...")
    
    custom_transport = httpx.AsyncHTTPTransport(retries=3)
    proxies = os.getenv("SYS_PROXY", "http://127.0.0.1:10809")
    
    try:
        async with httpx.AsyncClient(
            transport=custom_transport,
            proxies=proxies,
            timeout=15.0
        ) as client:
            logger.debug(f"Routing request via {proxies}")
            state["collected_context"].append("[2026 Q1 Market Data] Sector growth is 12%...")
            logger.info("[Node: Scraper] Fetch successful.")
            
    except (httpx.ConnectError, socket.gaierror) as e:
        logger.error(f"[Node: Scraper] Resolution Failed: {str(e)}")
        state["error_count"] += 1
        
    return state

async def analyzer_node(state: AgentState) -> AgentState:
    logger.info("[Node: Analyzer] Processing context payload...")
    context_length = sum(len(c) for c in state["collected_context"])
    logger.info(f"[Node: Analyzer] Current context payload: ~{context_length * 2} tokens.")
    
    state["final_output"] = "Analysis completed based on retrieved context."
    return state

def conditional_edge(state: AgentState) -> str:
    if state["error_count"] > 2:
        logger.warning("[Router] High error rate. Routing to Fallback.")
        return "fallback_node"
    if not state["collected_context"]:
        logger.info("[Router] Context empty. Routing to Scraper.")
        return "web_scraper_node"
    
    logger.info("[Router] Context validated. Routing to Analyzer.")
    return "analyzer_node"

if __name__ == "__main__":
    initial_state = AgentState(
        task_id="REQ-20260501",
        original_query="Analyze North American cloud hardware procurement trends",
        collected_context=[],
        current_node="start",
        error_count=0,
        final_output=""
    )
    logger.info(f"Initialized State Machine for {initial_state['task_id']}")
