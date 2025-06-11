from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any
import os

class FileToolInputs(BaseModel):
    file_path: str = Field(..., description="The path to the file.")
    content: str = Field(None, description="The content to write to the file.")

class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Writes content to a specified file. Use this to save control parameters and reasoning."
    args_schema: Type[BaseModel] = FileToolInputs

    def _run(self, file_path: str, content: str = "") -> str:
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            return f"File {file_path} written successfully."
        except Exception as e:
            return f"Error writing file: {e}"

class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Reads the content of a specified file. Use this to read control options and previous results."
    args_schema: Type[BaseModel] = FileToolInputs

    def _run(self, file_path: str, **kwargs: Any) -> str:
        try:
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}"
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}" 