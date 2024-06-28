from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from typing import List

class SearchQuery(BaseModel):
    query: str = Field(description="The search query to be executed")
    query_en: str = Field(description="The search query in English")

class SearchQueries(BaseModel):
    queries: List[SearchQuery] = Field(description="The search queries to be executed")