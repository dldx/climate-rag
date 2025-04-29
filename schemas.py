from typing import Any, List, Literal, Optional, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class SearchQuery(BaseModel):
    query: str = Field(description="The search query to be executed")
    query_en: str = Field(description="The search query in English")


class SearchQueries(BaseModel):
    queries: List[SearchQuery] = Field(description="The search queries to be executed")


class SourceMetadata(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description="A relevant title of the report, webpage, or document. This should be something unique that references the particular product name or article title, and not just a company name. Try to create a specific title. Include the name of the product, company, or article, and any other relevant information.",
    )
    company_name: Optional[str] = Field(
        default=None, description="The name of the company that published the document"
    )
    publishing_date: Optional[Tuple[int, Optional[int], Optional[int]]] = Field(
        default=None,
        description="The date the document was published, in the format (year, month, day)",
    )
    key_entity: Optional[str] = Field(
        default=None,
        description="The most important entity mentioned in the document. An entity is a person, place, or thing that is relevant to the document. For example, if the document is about a company, the key entity might be the company name. If the document is about a person, the key entity might be the person's name. If the document is about a place, the key entity might be the place name. If the document is about a product, the key entity might be the product name.",
    )
    key_entities: List[str] = Field(
        description="The 10 most important entities mentioned in the document. An entity is a person, place, or thing that is relevant to the document. For example, if the document is about a company, the key entities might be the company name. If the document is about a person, the key entities might be the person's name. If the document is about a place, the key entities might be the place name. If the document is about a product, the key entities might be the product and company names. If the document is about a news event, the key entities might be the people, places, and things involved in the event. Entities should be proper nouns, not general concepts. There should be no more than 10 entities.",
    )
    type_of_document: str = Field(
        description="The type of document, such as a 'report', 'blog post', 'news article', 'product page', 'index/store/gallery page'. If the document type is not in the list, return 'other'"
    )
    keywords: List[str] = Field(
        description="10 keywords that describe the topics in the document."
    )
    self_published: bool = Field(
        description="Whether the document is self-published by the company or government entity mentioned in the document. Ie. the document is published on the company's or government's own website or written as a first-party source."
    )
    primary_language: Optional[str] = Field(
        default="en",
        description="The primary language of the document. Use the ISO 639-1 language code (e.g., 'en' for English, 'zh' for Chinese, 'vi' for Vietnamese).",
    )


class PDFMetadata(SourceMetadata):
    num_pages: int = Field(description="The number of pages in the PDF document")
    scanned_pdf: bool = Field(
        description="Whether the PDF is a scanned document (True) or a digital one (False)"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
    - llm: The language model to use
    - initial_question: The initial question asked by the user
    - question: The question asked by the user
    - question_en: The question asked by the user in English
    - search_prompts: The search prompts to use
    - search_query: The search query to use
    - search_query_en: The search query to use in English
    - max_search_queries: The maximum number of search queries to use
    - generation: The generation to use
    - web_search: The web search to use
    - web_search_completed: Whether the web search has been completed
    - documents: The documents to use
    - user_happy_with_answer: Whether the user is happy with the answer
    - rag_filter: The RAG filter to use
    - shall_improve_question: Whether we shall improve the question
    - do_rerank: Whether we shall rerank the documents
    - do_crawl: Whether we shall crawl the web
    - do_add_additional_metadata: Whether we shall add additional metadata
    - search_tool: The search tool to use, either "serper", "tavily", or "baidu"
    - language: The language to use, in two-letter ISO 639-1 format. eg. "en", "zh", "vi"
    - initial_generation: Whether this is the initial generation
    - history: The history of the graph
    - mode: The mode to use when outputting the graph, either "gui" or "cli"
    - qa_id: The relevant Q&A ID for the question/answer pair
    - happy_with_answer: Whether the user is happy with the answer

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
