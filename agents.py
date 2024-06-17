from datetime import datetime, timezone
from typing import Any, List, Literal
from langchain.schema import Document
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from tools import (
    web_search_tool,
    add_urls_to_db,
    add_urls_to_db_firecrawl,
    format_docs,
    enc
)
import numpy as np

import os
from dotenv import load_dotenv

load_dotenv()

run_local = False
local_llm = "mistral"
MAX_TOKEN_LENGTH = 24_000


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    search_prompts: str
    search_query: str
    generation: str
    web_search: str
    web_search_completed: bool
    documents: List[Document]
    user_happy_with_answer: bool
    db: Any
    rag_filter: str
    shall_improve_question: bool
    do_rerank: bool
    search_tool: Literal["serper", "tavily", "baidu"]
    language: Literal["en", "zh", "vi"]


def get_current_utc_datetime():
    now_utc = datetime.now(timezone.utc)
    current_time_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
    return current_time_utc


def improve_question(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    ### Question Re-writer

    # LLM
    if run_local:
        llm = ChatOllama(model=local_llm, temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-4o")

    # Prompt
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n
        for vectorstore retrieval. Look at the initial and formulate an improved question. \n
        Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
        input_variables=["generation", "question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    # question_rewriter.invoke({"question": question})

    # Re-write question
    if state["shall_improve_question"]:
        better_question = question_rewriter.invoke({"question": state["question"]})
        print("---TRANSFORM QUERY---")
        print("Better Question: ", better_question)
        state["question"] = better_question
    return {"question": state["question"]}


def formulate_query(state: GraphState):
    """
    Formulate a query for RAG and web search based on the user question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, question, that contains search query
    """
    print("---FORMULATE QUERY---")
    question = state["question"]

    ### Convert question into search query
    from prompts import planning_agent_prompt

    # LLM
    if False:  # run_local == "Yes":
        llm = ChatOllama(model=local_llm, temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-4o")

    # Prompt
    create_search_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                planning_agent_prompt.format(
                    plan=None, feedback=None, datetime=get_current_utc_datetime()
                ),
            ),
            ("human", "{question}"),
        ]
    )

    search_prompt_creator = create_search_prompt | llm | StrOutputParser()
    search_prompt = search_prompt_creator.invoke({"question": question})

    state["search_prompts"] = search_prompt

    return {"search_prompts": search_prompt}


from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class SearchQuery(BaseModel):
    query: str = Field(description="The search query to be executed")
    query_en: str = Field(description="The search query in English")


def generate_search_query(state: GraphState):
    from prompts import generate_searches_prompt
    from langcodes import Language

    # LLM
    if False:  # run_local == "Yes":
        llm = ChatOllama(model=local_llm, temperature=0, format="json")
    else:
        llm = ChatOpenAI(model="gpt-4o")

    parser = PydanticOutputParser(pydantic_object=SearchQuery)

    extract_search_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generate_searches_prompt),
            ("human", "Query: {question}\n\nPlan: {search_prompts}"),
        ]
    ).partial(
        response_format=parser.get_format_instructions(),
        datetime=get_current_utc_datetime(),
        language=Language.get(state["language"]).display_name(),
    )

    extract_search_chain = extract_search_prompt | llm | parser
    extract_search = extract_search_chain.invoke(
        {"question": state["question"], "search_prompts": state["search_prompts"]}
    )

    state["search_query"] = extract_search.query
    print("Search Query: ", state["search_query"])
    print("Search Query (English): ", extract_search.query_en)

    return {"search_query": state["search_query"]}


def retrieve(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    db = state["db"]

    # Retrieval
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 25})
    documents = retriever.invoke(state["search_query"])
    if state["rag_filter"] is not None:
        documents = [doc for doc in documents if state["rag_filter"] in doc.metadata["source"]]
    state["documents"] = documents
    print("Retrieved Documents: ", [doc.metadata["id"] for doc in documents])
    return {"documents": documents}

def rerank_docs(state: GraphState):
    """
    Rerank documents using Cohere API

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with reranked documents
    """

    print("---RERANK DOCUMENTS---")
    documents = state["documents"]
    question = state["question"]

    if state["do_rerank"]:
        # Cohere API
        import cohere

        co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        reranked_results = co.rerank(model="rerank-multilingual-v3.0", query=question, documents=[doc.page_content for doc in documents])
        state["documents"] = [documents[docIndex.index] for docIndex in reranked_results.results if docIndex.relevance_score > 0.05]
        print("Reranked Documents: ", [doc.metadata["id"] for doc in state["documents"]])
    return {"documents": state["documents"]}

def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    if local_llm:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    else:
        llm = ChatOllama(model=local_llm, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llm | JsonOutputParser()

    # Score each doc - don't score web search results
    filtered_docs = list(filter(lambda x: x.metadata.get("web_search"), documents))
    for d in filter(lambda x: not x.metadata.get("web_search"), documents):
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    state["search_web"] = len(filtered_docs) <= 3
    print("Filtered Docs: ", filtered_docs)
    print("Search Web: ", state["search_web"])
    state["documents"] = filtered_docs
    return state


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    from langchain_community.utilities import GoogleSerperAPIWrapper
    from serpapi import BaiduSearch

    search_tool = state["search_tool"]  # "serper" or "tavily" or "baidu"

    print("---WEB SEARCH---")
    search_query = state["search_query"]
    print("Search Query: ", search_query)
    documents = state["documents"]

    # Web search
    if search_tool == "serper":
        search = GoogleSerperAPIWrapper(gl="gb")
        docs = search.results(search_query)["organic"]
        web_results = [
            Document(
                page_content=doc["snippet"],
                metadata={"source": doc["link"], "web_search": True},
            )
            for doc in docs
        ]
    elif search_tool == "tavily":
        docs = web_search_tool.invoke({"query": search_query})
        web_results = [
            Document(
                page_content=doc["content"],
                metadata={"source": doc["url"], "web_search": True},
            )
            for doc in docs
        ]
    elif search_tool == "baidu":
        search = BaiduSearch({"q": search_query, "api_key": os.getenv("SERPAPI_API_KEY")})
        docs = search.get_dict()["organic_results"]
        web_results = [
            Document(
                page_content=doc.get("snippet", ""),
                metadata={"source": doc["link"], "web_search": True},
            )
            for doc in docs
        ]
    documents += web_results[:10]

    return {"documents": documents}


def crawl_or_not(doc: Document, question: str):
    from prompts import crawl_grader_prompt

    """Decide whether to crawl a web page or not, and if so, which urls to scrape.

    Args:
        docs: A list of documents to evaluate.
        question: The question to answer.

    Returns:
        A dictionary with the following format:
        {
            "crawl": bool,
            "urls_to_scrape": List[str]
        }
    """

    llm = ChatOpenAI(model="gpt-4o")
    # llm = ChatOllama(model=local_llm, temperature=0, format="json")

    class CrawlDecision(BaseModel):
        crawl: bool = Field(description="Whether the web page should be crawled")
        urls_to_scrape: List[str] = Field(
            description="The urls that should be scraped if the web page should be crawled"
        )

    parser = PydanticOutputParser(pydantic_object=CrawlDecision)

    crawl_grader_message = ChatPromptTemplate.from_messages(
        [
            ("system", crawl_grader_prompt),
            (
                "human",
                """Web page contents: {context}

    URL: {url}

    Question: {question}""",
            ),
        ]
    ).partial(response_format=parser.get_format_instructions())

    crawl_grader = crawl_grader_message | llm | parser

    context = doc.page_content
    # Length of context
    context_length = len(enc.encode(context))
    if context_length > MAX_TOKEN_LENGTH:
        print("Length of context too long, truncating: ", len(enc.encode(context)))
        print("Original Length: ", len(context))
        # Reduce context length by ratio
        reduced_size = int(
            (context_length - MAX_TOKEN_LENGTH) / MAX_TOKEN_LENGTH * len(context)
        )
        context = context[:-reduced_size]
        print("Reduced Length: ", len(context), len(enc.encode(context)))

    crawl_decision = crawl_grader.invoke(
        {
            "context": context,
            "question": question,
            "url": (
                doc.metadata["sourceURL"]
                if "sourceURL" in doc.metadata.keys()
                else doc.metadata["source"]
            ),
        }
    )

    return crawl_decision


def add_urls_to_database(state):
    """
    Add web search results to the database.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---ADD WEB SEARCH RESULTS TO DATABASE---")
    documents = state["documents"]
    db = state["db"]

    # Add web search results to database
    # Filter only web search results
    use_firecrawl = True
    if use_firecrawl:
        # Get list of urls
        urls = list(
            map(
                lambda x: x.metadata["source"],
                filter(lambda x: x.metadata.get("web_search") == True, documents),
            )
        )
        # Add urls to database
        docs = add_urls_to_db_firecrawl(urls, db)
        # Check if docs need to be further crawled
        for doc in docs:
            crawl_decision = crawl_or_not(doc, state["question"])
            print(crawl_decision)
            if crawl_decision.crawl:
                if len(crawl_decision.urls_to_scrape) > 10:
                    print("Too many urls to scrape, only scraping first 10")
                add_urls_to_db_firecrawl(
                    list(
                        filter(
                            lambda x: True,  # "fjord-coffee.de" in x,
                            crawl_decision.urls_to_scrape,
                        )
                    )[:10],
                    db,
                )
    else:
        # Get list of urls
        urls = list(
            map(
                lambda x: "https://r.jina.ai/" + x.metadata["source"],
                filter(
                    lambda x: x.metadata.get("web_search") == True, state["documents"]
                ),
            )
        )
        # Add urls to database
        add_urls_to_db(urls, db)
    state["web_search_completed"] = True

    return {"web_search_completed": state["web_search_completed"]}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or run a web search first.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    # if state.get("web_search_completed", False):
    return "generate"
    # else:
    #     return "web_search"


def ask_user_for_feedback(state):
    """
    Asks the user if they are happy with the generated answer or if we should search the web.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call

    """
    user_decision = input("Are you happy with the answer? (y/n)").lower()[0] == "y"
    state["user_happy_with_answer"] = user_decision

    return {
        "user_happy_with_answer": user_decision,
        "documents": state["documents"],
        "generation": state["generation"],
    }


def decide_to_search(state):
    """
    Asks the user if they are happy with the generated answer or if we should search the web.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call

    """

    if state["user_happy_with_answer"]:
        return "END"
    else:
        state.pop("user_happy_with_answer", None)
        return "web_search"


def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    from prompts import generate_prompt

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Get length of document tokens, and filter so that total tokens is less than 30,000
    documents = np.array(documents)[
        np.array([len(enc.encode(doc.page_content)) for doc in documents]).cumsum()
        < MAX_TOKEN_LENGTH
    ].tolist()

    prompt = PromptTemplate(
        template=generate_prompt,
        input_variables=["question", "context"],
    )

    # LLM
    if run_local:
        llm = ChatOllama(model=local_llm, temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-4o")

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    context = format_docs(documents)

    # Length of context
    context_length = len(enc.encode(context))
    if context_length > MAX_TOKEN_LENGTH:
        print("Length of context too long, truncating: ", len(enc.encode(context)))
        print("Original Length: ", len(context))
        # Reduce context length by ratio
        reduced_size = int(
            (context_length - MAX_TOKEN_LENGTH) / MAX_TOKEN_LENGTH * len(context)
        )
        context = context[:-reduced_size]
        print("Reduced Length: ", len(context), len(enc.encode(context)))

    # RAG generation
    generation = rag_chain.invoke({"context": context, "question": question})

    state["generation"] = generation
    state["documents"] = documents

    return {"generation": generation, "documents": documents}
