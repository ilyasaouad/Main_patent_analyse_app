"""
LangGraph workflow orchestrator for patent analysis system.
Connects all agents in a multi-agent workflow for comprehensive patent examination assistance.
"""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from loguru import logger

from config import PatentAnalysisState, settings
from agents import (
    DocumentReaderAgent,
    ClaimsAnalystAgent,
    PriorArtSearchAgent,
    NoveltyAgent,
    InventiveStepAgent,
    IndustrialApplicabilityAgent,
    InfringementAgent,
    ReportGeneratorAgent
)

def create_patent_workflow():
    """
    Initializes the LangGraph workflow with all 8 agents.
    """
    workflow = StateGraph(PatentAnalysisState)

    # 1. Initialize Agents
    doc_reader = DocumentReaderAgent()
    # claims_analyst = ClaimsAnalystAgent()
    # ... other agents on hold

    # 2. Add Nodes
    workflow.add_node("document_reader", lambda state: {"extracted_text": doc_reader.run(state.document_path)[0]})
    # workflow.add_node("claims_analyst", claims_analyst.run)
    # ... other nodes on hold

    # 3. Define Edges
    workflow.set_entry_point("document_reader")
    workflow.add_edge("document_reader", END)

    return workflow.compile()

# Singleton instance of the compiled graph
patent_analyzer = create_patent_workflow()

if __name__ == "__main__":
    logger.info("Patent Workflow Orchestrator initialized.")
