from datetime import datetime, timezone
import msgspec
import pandas as pd
from helpers import generate_qa_id
from schemas import GraphState, SearchQuery, SearchQueries
from typing import Any, List, Literal
from typing_extensions import TypedDict
from langcodes import Language
from langchain_core.output_parsers import StrOutputParser

from langchain.schema import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from llms import get_chatbot, get_max_token_length
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from tools import (
    get_sources_based_on_filter,
    web_search_tool,
    get_vector_store,
    add_urls_to_db,
    add_urls_to_db_firecrawl,
    format_docs,
    enc,
)
import numpy as np

import os
from dotenv import load_dotenv

from cache import r

load_dotenv()


def get_current_utc_datetime():
    now_utc = datetime.now(timezone.utc)
    current_time_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
    return current_time_utc


class ImprovedQuestion(BaseModel):
    question: str = Field(description="The improved question")
    question_en: str = Field(description="The improved question in English")


def improve_question(state: GraphState) -> GraphState:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    from prompts import question_rewriter_prompt

    ### Question Re-writer

    # LLM
    llm = get_chatbot(state["llm"])

    parser = PydanticOutputParser(pydantic_object=ImprovedQuestion)

    # Prompt
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_rewriter_prompt),
            ("human", "Initial question: {question}"),
        ]
    ).partial(
        question=state["question"],
        response_format=parser.get_format_instructions(),
        datetime=get_current_utc_datetime(),
        language=Language.get(state["language"]).display_name(),
    )

    question_rewriter = re_write_prompt | llm | parser

    state["initial_question"] = state["question"]
    # Re-write question
    if state["shall_improve_question"]:
        better_question = question_rewriter.invoke({"question": state["question"]})
        print("---TRANSFORM QUERY---")
        state["question"] = better_question.question
        state["question_en"] = better_question.question_en
        print("Better Question: ", state["question"])
        if state["question_en"] != state["question"]:
            print("Better Question (English): ", state["question_en"])
    return {
        "question": state["question"],
        "question_en": state["question_en"],
        "initial_question": state["initial_question"],
    }


from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


def formulate_query(state: GraphState) -> GraphState:
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

    n_queries = max(state["max_search_queries"], 5)

    llm = get_chatbot(
        state["llm"], model_kwargs={"response_format": {"type": "json_object"}}
    )

    parser = PydanticOutputParser(pydantic_object=SearchQueries)

    # Prompt
    create_search_prompts = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                planning_agent_prompt,
            ),
            ("human", "{question}"),
        ]
    ).partial(
        response_format=parser.get_format_instructions(),
        n_queries=n_queries,
        plan=None,
        feedback=None,
        datetime=get_current_utc_datetime(),
        language=Language.get(state["language"]).display_name(),
    )

    search_prompt_creator = create_search_prompts | llm | parser
    search_prompts = search_prompt_creator.invoke({"question": question})

    state["search_prompts"] = search_prompts.queries

    return {"search_prompts": state["search_prompts"]}


def generate_search_query(state: GraphState) -> GraphState:
    from prompts import generate_searches_prompt

    llm = get_chatbot(
        state["llm"], model_kwargs={"response_format": {"type": "json_object"}}
    )

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
    state["search_query_en"] = extract_search.query_en
    print("Search Query: ", state["search_query"])
    if state["search_query_en"] != state["search_query"]:
        print("Search Query (English): ", state["search_query_en"])

    return {
        "search_query": state["search_query"],
        "search_query_en": state["search_query_en"],
    }


def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    print("Search Queries: ", state["search_prompts"])
    db = get_vector_store()

    ## Retrieve docs
    # 100 docs max
    k = 100
    # List of source documents that we want to include
    rag_filter = state.get("rag_filter", None)
    if rag_filter == "":
        rag_filter = None
    if rag_filter is not None:
        source_list = get_sources_based_on_filter(rag_filter, r)
        if len(source_list) == 0:
            print("No source documents found in database")
            return {"documents": [], "search_prompts": state["search_prompts"]}
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": {"source": {"$in": source_list}}},
        )
    else:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # We will retrieve docs based on many queries
    from tools import retrieve_multiple_queries

    documents = retrieve_multiple_queries(
        [
            (query.query_en if state["language"] == "en" else query.query)
            for query in state["search_prompts"]
        ],
        retriever=retriever,
        k=k,
    )

    state["documents"] = documents
    print("Retrieved Documents: ", [doc.metadata["id"] for doc in documents])
    return {"documents": documents, "search_prompts": state["search_prompts"]}


from tools import get_source_document_extra_metadata
from functools import partial


def get_metadata_for_source(r, use_llm: bool, source: str) -> dict:
    try:
        metadata = get_source_document_extra_metadata(
            r, source, metadata_fields=["title", "company_name", "publishing_date"], use_llm=use_llm
        )
    except:
        # If metadata not found, continue
        metadata = {}
    metadata["source"] = source
    return metadata


def add_additional_metadata(state: GraphState) -> GraphState:
    """
    Add additional metadata to documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with additional metadata
    """

    print("---ADD ADDITIONAL METADATA TO DOCUMENTS---")
    documents = state["documents"]

    # Add additional metadata to documents

    # Get unique sources
    unique_sources = list(set([doc.metadata.get("source", None) for doc in documents]))
    # For each source, get titles and company names (if available)
    # Use partial to pass the 'r' argument to the function
    func = partial(get_metadata_for_source, r, state["do_add_additional_metadata"])
    source_metadatas = list(map(func, unique_sources))

    # Now add metadata to documents
    source_metadatas_df = pd.DataFrame(source_metadatas).reindex(["source", "title", "company_name", "publishing_date"], axis=1).set_index("source")
    for doc in documents:
        doc.metadata = {
            **doc.metadata,
            **source_metadatas_df.loc[doc.metadata["source"]],
        }

    state["documents"] = documents
    return {"documents": state["documents"]}


def rerank_docs(state: GraphState) -> GraphState:
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
    formatted_docs = [
            f"""Title: {doc.metadata.get("title", "") if not pd.isna(doc.metadata.get("title", "")) else ""}
Company name: '{doc.metadata.get("company_name", "") if not pd.isna(doc.metadata.get("company_name", "")) else ""}'
Contents: {doc.page_content}
"""
            for doc in documents
        ]

    rerank_api = "cohere"

    if rerank_api == "jina":
        from langchain_community.document_compressors import JinaRerank
        try:
            reranker = JinaRerank(model="jina-reranker-v2-base-multilingual")
            reranked_results = reranker.rerank(query=question, documents=formatted_docs, top_n=len(documents))
        except Exception as e:
            print("Reranker failed: ", e)
            rerank_api = "cohere"
        state["documents"] = [
            documents[docIndex["index"]]
            for docIndex in reranked_results
            if docIndex["relevance_score"] > 0.05
        ]
    if rerank_api == "cohere":
        # Cohere API
        import cohere

        co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

        try:
            reranked_results = co.rerank(
                model="rerank-multilingual-v3.0",
                query=question,
                documents=formatted_docs,
            )
            state["documents"] = [
                documents[docIndex.index]
                for docIndex in reranked_results.results
                if docIndex.relevance_score > 0.05
            ]
        except Exception as e:
            print("Reranker failed: ", e)
    print(
        "Reranked Documents: ", [doc.metadata["id"] for doc in state["documents"]]
    )

    return {"documents": state["documents"]}


def grade_documents(state: GraphState) -> GraphState:
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

    llm = get_chatbot(
        llm=state["llm"],
        model_kwargs={"response_format": {"type": "json_object"}},
    )

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


def web_search(state: GraphState) -> GraphState:
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
    num_queries = state["max_search_queries"]  # How many search queries to use

    print("---WEB SEARCH---")
    print(f"Running {num_queries} search queries")
    documents = state["documents"] or []
    search_results = []
    for search_prompt in state["search_prompts"][:num_queries]:
        if state["language"] == "en":
            search_query = search_prompt.query_en
        else:
            search_query = search_prompt.query

        print("Search query: ", search_query)
        if state["language"] != "en":
            print("Search query (en): ", state["search_prompts"][0].query_en)

        # Web search
        if search_tool == "serper":
            search = GoogleSerperAPIWrapper(gl="gb")
            docs = search.results(search_query)["organic"]
            web_results = [
                Document(
                    page_content=doc.get("snippet", ""),
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
            search = BaiduSearch(
                {"q": search_query, "api_key": os.getenv("SERPAPI_API_KEY")}
            )
            docs = search.get_dict()["organic_results"]
            web_results = [
                Document(
                    page_content=doc.get("snippet", ""),
                    metadata={"source": doc["link"], "web_search": True},
                )
                for doc in docs
            ]
        search_results += [
            {"source": doc.metadata["source"], "rank": i, "doc": doc}
            for i, doc in enumerate(web_results)
        ]
    # Rank results by frequency in search results and rank in individual search results
    # Use average of combined rank and frequency rank to sort
    ranked_search_results = pd.DataFrame(search_results).pipe(lambda df: df.join(df.value_counts("source"), on="source")).assign(final_rank = lambda x: x["rank"] - x["count"]).groupby("source").agg({"final_rank": "mean", "doc":"first"}).sort_values("final_rank").doc.tolist()
    # Scale total number of results by number of queries
    max_results_to_add = min((5 * num_queries), 20)
    # Add web search results to documents
    documents += ranked_search_results[:max_results_to_add]

    state["documents"] = documents

    print(f"""Identified {len(documents)} web search results: {
        [doc.metadata["source"] for doc in documents if doc.metadata.get("web_search")]
    }""")

    return {
        "documents": documents,
        "search_prompts": state["search_prompts"],
        "search_query": search_query,
        "search_query_en": state["search_prompts"][0].query_en,
    }


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

    llm = get_chatbot(llm="gpt-4o-mini")
    MAX_TOKEN_LENGTH = get_max_token_length("gpt-4o-mini")

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


def add_urls_to_database(state: GraphState) -> GraphState:
    """
    Add web search results to the database.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---ADD WEB SEARCH RESULTS TO DATABASE---")
    documents = state["documents"]
    db = get_vector_store()

    # Add web search results to database
    # Filter only web search results
    use_firecrawl = False
    # Get list of urls
    urls = list(
        map(
            lambda x: x.metadata["source"],
            filter(lambda x: x.metadata.get("web_search") == True, documents),
        )
    )
    # Add urls to database
    docs = add_urls_to_db(urls, db, use_firecrawl=use_firecrawl)
    # Check if docs need to be further crawled
    if state["do_crawl"]:
        for doc in docs:
            crawl_decision = crawl_or_not(doc, state["question"])
            print(crawl_decision)
            if crawl_decision.crawl:
                if len(crawl_decision.urls_to_scrape) > 10:
                    print("Too many urls to scrape, only scraping first 10")
                add_urls_to_db(
                    list(
                        filter(
                            lambda x: True,
                            crawl_decision.urls_to_scrape,
                        )
                    )[:10],
                    db,
                    use_firecrawl=use_firecrawl,
                )
    state["web_search_completed"] = True

    return {"web_search_completed": state["web_search_completed"]}


### Edges
def decide_to_rerank(state: GraphState) -> GraphState:
    """
    Determines whether to rerank documents

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    if state.get("do_rerank", False) and (len(state["documents"]) > 1):
        return "rerank"
    else:
        return "no_rerank"


def decide_to_add_additional_metadata(state: GraphState) -> GraphState:
    """
    Determines whether to add additional metadata to documents

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    if state.get("do_add_additional_metadata", False) and (len(state["documents"]) > 0):
        return "add_additional_metadata"
    else:
        return "no_add_additional_metadata"


def ask_user_for_feedback(state: GraphState) -> GraphState:
    """
    Asks the user if they are happy with the generated answer or if we should search the web.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call

    """
    user_decision = state["happy_with_answer"]

    return {
        "user_happy_with_answer": user_decision,
        "documents": state["documents"],
        "generation": state["generation"],
    }


def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, or skip to web search.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    if state["initial_generation"]:
        return "generate"
    else:
        # Change initial_generation to True for next cycle
        state["initial_generation"] = True
        return "no_generate"


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


def generate(state: GraphState) -> GraphState:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    from prompts import generate_prompt

    print("---GENERATE---")
    documents = state["documents"]
    llm = get_chatbot(state["llm"])
    MAX_TOKEN_LENGTH = get_max_token_length(state["llm"])


    # If there are no documents, save a message instead
    if len(documents) == 0:
        generation = (
            """No relevant documents found to generate answer from!"""
            + (" Check your RAG filters are correct!" if state["rag_filter"] else "")
        )
    else:
        # Get length of document tokens, and filter so that total tokens is less than 30,000
        documents = np.array(documents)[
            np.array([len(enc.encode(doc.page_content)) for doc in documents]).cumsum()
            < MAX_TOKEN_LENGTH
        ].tolist()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", generate_prompt),
                ("human", """Question: {question}

Context:
{context}

A reminder of the question you should answer: {question}

Remember to return the answer in English only.""",
                ),
            ]
        )

        # LLM Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Format context
        context = format_docs(documents)

        # Calculate length of context
        context_length = len(enc.encode(context))
        # Reduce context if too long to fit within token window
        if context_length > MAX_TOKEN_LENGTH:
            print("Length of context too long, truncating: ", len(enc.encode(context)))
            print("Original Length: ", len(context))
            # Reduce context length by ratio
            reduced_size = int(
                (context_length - MAX_TOKEN_LENGTH) / MAX_TOKEN_LENGTH * len(context)
            )
            context = context[:-reduced_size]
            print("Reduced Length: ", len(context), len(enc.encode(context)))

        # Generate answer given context and question
        generation = rag_chain.invoke(
            {"context": context, "question": state["question_en"]}
        )

    # Save generated answer to cache
    qa_id = generate_qa_id(state["initial_question"], generation)
    qa_map = {
        "answer": generation,
        "question": state["initial_question"],
        "rag_filter": state["rag_filter"] if state["rag_filter"] else "*",
        "doc_ids": msgspec.json.encode([doc.metadata["id"] for doc in documents]),
        "sources": msgspec.json.encode(
            list(
                set(
                    [
                        (
                            doc.metadata["source"]
                            if "source" in doc.metadata.keys()
                            else ""
                        )
                        for doc in documents
                    ]
                )
            )
        ),
        "date_added": int(datetime.timestamp(datetime.now(timezone.utc))),
    }
    r.hset(f"climate-rag::answer:{qa_id}", mapping=qa_map)

    state["generation"] = generation
    state["documents"] = documents
    state["qa_id"] = qa_id

    return state
