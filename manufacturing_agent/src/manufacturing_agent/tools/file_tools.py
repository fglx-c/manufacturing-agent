from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any
import os

# In-memory storage acting as a virtual filesystem for CrewAI tools
MEM_STORE: dict[str, str] = {}

class FileToolInputs(BaseModel):
    file_path: str = Field(..., description="The path to the file.")
    content: str = Field(None, description="The content to write to the file.")

class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Writes content to a specified file. Use this to save control parameters and reasoning."
    args_schema: Type[BaseModel] = FileToolInputs

    def _run(self, file_path: str, content: str = "") -> str:
        """Cache *content* in the in-memory MEM_STORE instead of the real FS."""
        MEM_STORE[file_path] = content
        return f"Content cached for {file_path}."

class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Reads the content of a specified file. Use this to read control options and previous results."
    args_schema: Type[BaseModel] = FileToolInputs

    def _run(self, file_path: str, **kwargs: Any) -> str:
        """Return cached content from MEM_STORE or an error string."""
        return MEM_STORE.get(file_path, f"Error: File not found at {file_path}") 