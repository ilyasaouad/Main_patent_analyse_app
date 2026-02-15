"""
Configuration settings for the Patent Analysis Orchestrator.
"""

import os
from pathlib import Path

class Settings:
    """Application configuration settings."""
    
    # Application Info
    APP_NAME = "Patent Analysis Orchestrator"
    APP_VERSION = "1.0.0"
    
    # Document Processing
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "200"))  # MB
    SUPPORTED_FORMATS = [".pdf", ".docx"]
    
    # Directories
    BASE_DIR = Path(__file__).parent.parent
    TEMP_DIR = BASE_DIR / "temp"
    LOGS_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [cls.TEMP_DIR, cls.LOGS_DIR, cls.DATA_DIR, cls.OUTPUT_DIR]:
            directory.mkdir(exist_ok=True)

# Initialize settings
settings = Settings()
settings.ensure_directories()
