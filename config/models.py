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
    document_path: str = ""
    document_info: Optional[DocumentInfo] = None
    extracted_text: str = ""
    
    # Analysis results (to be expanded)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    
    # Workflow control
    current_agent: str = "document_reader"
    processing_complete: bool = False
