import datetime
import pandas as pd
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from pydantic import field_serializer
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document

from typing import Any, List, Literal, Optional, Tuple

class SearchQuery(BaseModel):
    query: str = Field(description="The search query to be executed")
    query_en: str = Field(description="The search query in English")

class SearchQueries(BaseModel):
    queries: List[SearchQuery] = Field(description="The search queries to be executed")

class SourceMetadata(BaseModel):
    title: Optional[str] = Field(default=None, description="A relevant title of the report, webpage, or document. This should be something unique that references the particular product name or article title, and not just a company name. Try to create a specific title. Include the name of the product, company, or article, and any other relevant information.")
    company_name: Optional[str] = Field(default=None, description="The name of the company that published the document")
    publishing_date: Optional[Tuple[int, Optional[int], Optional[int]]] = Field(default=None, description="The date the document was published, in the format (year, month, day)")
    key_entity: Optional[str] = Field(default=None, escription="The most important entity mentioned in the document. An entity is a person, place, or thing that is relevant to the document. For example, if the document is about a company, the key entity might be the company name. If the document is about a person, the key entity might be the person's name. If the document is about a place, the key entity might be the place name. If the document is about a product, the key entity might be the product name.")
    key_entities: List[str] = Field(description="The 10 key entities mentioned in the document. An entity is a person, place, or thing that is relevant to the document. For example, if the document is about a company, the key entities might be the company name. If the document is about a person, the key entities might be the person's name. If the document is about a place, the key entities might be the place name. If the document is about a product, the key entities might be the product and company names. If the document is about a news event, the key entities might be the people, places, and things involved in the event. Entities should be proper nouns, not general concepts.")
    type_of_document: str = Field(description="The type of document, such as a 'report', 'blog post', 'news article', 'product page', 'index/store/gallery page', or other type of document")
    keywords: List[str] = Field(description="10 keywords that describe the topics in the document.")
    self_published: bool = Field(description="Whether the document is self-published by the company or government entity mentioned in the document. Ie. the document is published on the company's or government's own website or written as a first-party source.")



class PDFMetadata(SourceMetadata):
    num_pages: int = Field(description="The number of pages in the PDF document")
    scanned_pdf: bool = Field(description="Whether the PDF is a scanned document (True) or a digital one (False)")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    llm: Literal["gpt-4o", "gpt-3.5-turbo-16k", "mistral", "claude"]
    initial_question: str
    question: str
    question_en: str
    search_prompts: List[SearchQuery]
    search_query: str
    search_query_en: str
    max_search_queries: int
    generation: str
    web_search: str
    web_search_completed: bool
    documents: List[Document]
    user_happy_with_answer: bool
    rag_filter: str
    shall_improve_question: bool
    do_rerank: bool
    do_crawl: bool
    do_add_additional_metadata: bool
    search_tool: Literal["serper", "tavily", "baidu"]
    language: Literal["en", "zh", "vi"]
    initial_generation: bool
    history: List[Any]
    mode: Literal["gui", "cli"]
    qa_id: str
    happy_with_answer: bool

