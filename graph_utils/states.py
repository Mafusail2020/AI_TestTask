from pydantic import BaseModel
from typing import Dict, Optional, List, Any, Literal


class GenerateState(BaseModel):
    num_dialogues: int
    dialogues: Optional[Dict[str, Any]] = None
    output_file: Optional[str] = None

class AnalyzeState(BaseModel):
    input_file: str
    output_file: Optional[str] = None
    raw_dialogues: Optional[Dict[str, Any]] = None
    analyzed_dialogues: Optional[Dict[str, Any]] = None
