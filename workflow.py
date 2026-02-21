"""
LangGraph workflow orchestrator for patent analysis system.
Connects all agents in a multi-agent workflow for comprehensive patent examination assistance.
"""

from typing import Dict, Any, List, Optional
import os
from langgraph.graph import StateGraph, END
from loguru import logger

from config import PatentAnalysisState, settings
from agents import (
    DescriptionReaderSubAgent,
    ClaimsReaderSubAgent,
    DrawingReaderSubAgent,
    ClaimsAnalystAgent,
    # ... other agents
)


def create_patent_workflow():
    """
    Initializes the LangGraph workflow with separate readers for description, claims, and drawings.
    """
    workflow = StateGraph(PatentAnalysisState)

    # 1. Initialize Agents
    description_reader = DescriptionReaderSubAgent()
    claims_reader = ClaimsReaderSubAgent()
    drawing_reader = DrawingReaderSubAgent()

    # 2. Add Nodes
    def run_description(state):
        path = (
            state.description_path
            if hasattr(state, "description_path")
            else state.get("description_path")
        )
        text, _, abstract_text = description_reader.run(path)
        return {
            "description_text": text,
            "abstract_text": abstract_text,
            "current_agent": "claims_reader",
        }

    def run_claims(state):
        path = (
            state.claims_path
            if hasattr(state, "claims_path")
            else state.get("claims_path")
        )
        if path:
            text, _ = claims_reader.run(path)
        else:
            # Check if claims_text.txt was already populated by description_reader
            claims_file = "claims_text.txt"
            if os.path.exists(claims_file):
                with open(claims_file, "r", encoding="utf-8") as f:
                    text = f.read()
                if text.strip():
                    logger.info(f"Loaded claims from {claims_file}")
                else:
                    text = "Not provided."
            else:
                text = "Not provided."
        return {"claims_text": text, "current_agent": "drawing_reader"}

    def run_drawing(state):
        path = (
            state.drawings_path
            if hasattr(state, "drawings_path")
            else state.get("drawings_path")
        )
        if path:
            text, _ = drawing_reader.run(path)
        else:
            # Check if drawings_text.txt was already populated
            drawings_file = "drawings_text.txt"
            if os.path.exists(drawings_file):
                with open(drawings_file, "r", encoding="utf-8") as f:
                    text = f.read()
                if text.strip():
                    logger.info(f"Loaded drawings from {drawings_file}")
                else:
                    text = "Not provided."
            else:
                text = "Not provided."
        return {"drawings_text": text, "current_agent": "END"}

    workflow.add_node("description_reader", run_description)
    workflow.add_node("claims_reader", run_claims)
    workflow.add_node("drawing_reader", run_drawing)

    # 3. Define Edges
    workflow.set_entry_point("description_reader")
    workflow.add_edge("description_reader", "claims_reader")
    workflow.add_edge("claims_reader", "drawing_reader")
    workflow.add_edge("drawing_reader", END)

    return workflow.compile()


# Singleton instance of the compiled graph
patent_analyzer = create_patent_workflow()

if __name__ == "__main__":
    logger.info("Patent Workflow Orchestrator initialized.")
