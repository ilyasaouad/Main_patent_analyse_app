"""
Pydantic data models for patent analysis workflow.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class AgentStatus(str, Enum):
    """Status codes for agents."""

    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"


class DocumentInfo(BaseModel):
    """Metadata about the processed patent document."""

    filename: str
    file_size: int
    pages_count: Optional[int] = None
    text_extracted: bool = False
    ocr_required: bool = False
    file_path: str


class PatentAnalysisState(BaseModel):
    """
    Complete state for LangGraph workflow.
    """

    # File paths
    description_path: str = ""
    claims_path: str = ""
    drawings_path: str = ""
    abstract_path: str = ""

    # Extracted texts
    description_text: str = ""
    claims_text: str = ""
    drawings_text: str = ""
    abstract_text: str = ""

    # Document info (multiple now)
    description_info: Optional[DocumentInfo] = None
    claims_info: Optional[DocumentInfo] = None
    drawings_info: Optional[DocumentInfo] = None

    errors: List[str] = Field(default_factory=list)

    # Workflow control
    current_agent: str = "description_reader"
    processing_complete: bool = False
