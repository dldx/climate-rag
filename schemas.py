from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from typing import List, Optional, Tuple

class SearchQuery(BaseModel):
    query: str = Field(description="The search query to be executed")
    query_en: str = Field(description="The search query in English")

class SearchQueries(BaseModel):
    queries: List[SearchQuery] = Field(description="The search queries to be executed")
class PageMetadata(BaseModel):
    title: str = Field(description="A relevant title of the report, webpage, or document. This should be something unique that references the particular product name or article title, and not just a company name. Try to create a specific title. Include the name of the product, company, or article, and any other relevant information.")
    company_name: str = Field(description="The name of the company that published the document")
    publishing_date: Optional[Tuple[int, int, int]] = Field(default=None, description="The date the document was published, in the format (year, month, day)")
    key_entity: str = Field(description="The most important entity mentioned in the document. An entity is a person, place, or thing that is relevant to the document. For example, if the document is about a company, the key entity might be the company name. If the document is about a person, the key entity might be the person's name. If the document is about a place, the key entity might be the place name. If the document is about a product, the key entity might be the product name.")
    key_entities: List[str] = Field(description="The key entities mentioned in the document. An entity is a person, place, or thing that is relevant to the document. For example, if the document is about a company, the key entities might be the company name. If the document is about a person, the key entities might be the person's name. If the document is about a place, the key entities might be the place name. If the document is about a product, the key entities might be the product and company names. If the document is about a news event, the key entities might be the people, places, and things involved in the event. Entities should be specific nouns, not general concepts.")
    type_of_document: str = Field(description="The type of document, such as a 'report', 'blog post', 'news article', 'product page', 'index/store/gallery page', or other type of document")