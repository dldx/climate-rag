# api.py
import traceback
from typing import Annotated, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from cache import r, answer_index_name
from schemas import SourceMetadata
from helpers import compile_answer, get_previous_queries, humanize_unix_date
from tools import (
    get_sources_based_on_filter,
    clean_urls,
    get_source_document_extra_metadata,
)
import os
from redis.commands.search.query import Query as RedisQuery
import msgspec
import mistune
from datetime import datetime, timezone
import gradio as gr
from webapp import demo
import logging
from pymarkdown.api import PyMarkdownApi
from urllib.parse import urlencode

logging.getLogger("uvicorn").setLevel(logging.WARNING)

# Serve static files (like CSS, JS, images)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory=".")
templates.env.filters['urlencode'] = lambda x: urlencode(x) if x else ''


class SourceMetadataWithSource(SourceMetadata):
    source: str
    key_entities: Optional[List[str]] = None
    type_of_document: Optional[str] = None
    keywords: Optional[List[str]] = None
    self_published: Optional[bool] = None
    publishing_date: Optional[datetime] = None


class Answer(BaseModel):
    qa_id: str
    question: str
    answer: str
    sources: List[SourceMetadataWithSource]
    date_added: datetime
    date_added_ts: Optional[str] = ""


class Answers(BaseModel):
    results: List[Answer]


def _get_source_metadata(source: str, use_llm: bool = False):
    metadata = get_source_document_extra_metadata(
        r,
        source,
        metadata_fields=["title", "company_name", "publishing_date"],
        use_llm=use_llm,
    )
    return SourceMetadataWithSource(
        source=clean_urls(
            source,
            os.environ.get("STATIC_PATH", ""),
        )[0],
        **metadata,
    )


@app.get("/answers", response_model=Answers)
def get_answers(
    q: Optional[str] = Query(default=None, description="Search for questions"),
    limit: int = Query(
        default=10, ge=1, le=100, description="Limit the number of results"
    ),
    page_no: int = Query(default=1, description="Which page number of results to get"),
    include_metadata: bool = Query(
        default=False, description="Include document metadata"
    ),
):
    """
    Retrieve answers from Redis based on search terms.

    Args:
        q: (Optional) Search term to match against the question.
        source: (Optional) Search term to filter by source.
        limit: (Optional) Maximum number of results to return.
        page_no: (Optional) Page number of results to return.
        include_metadata: (Optional) Flag to include metadata in the response

    Returns:
        list: A list of dictionaries representing matching answers
    """
    if q is None:
        q = ""
    if "@" not in q:
        q = f"@question:({q})"

    # Get dataframe of previous queries
    df = get_previous_queries(
        r, query_filter=q, limit=limit, page_no=page_no, additional_fields=["sources"]
    )
    if include_metadata:
        df.sources = df.sources.apply(lambda x: map(_get_source_metadata, x))
    else:
        df.sources = df.sources.apply(
            lambda x: map(lambda y: SourceMetadataWithSource(source=y), x)
        )
    return Answers(results=df.to_dict("records"))


@app.get("/answers/{qa_id}", response_model=Answer)
def get_answer_by_id(
    qa_id: str,
    include_metadata: bool = Query(
        default=False, description="Include document metadata"
    ),
):
    """
    Retrieve a specific answer by its ID.

    Args:
        qa_id: The ID of the answer to retrieve.
        include_metadata: (Optional) Flag to include metadata in the response

    Returns:
        dict: A dictionary representing the matching answer, if found.
        Raises HTTPException if not found.
    """
    answer = r.hgetall(f"climate-rag::answer:{qa_id}")
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found")

    sources = msgspec.json.decode(answer["sources"]) if answer.get("sources") else []

    if include_metadata:
        sources = [_get_source_metadata(source) for source in sources]
    else:
        sources = [SourceMetadataWithSource(source=source) for source in sources]

    return Answer(
        qa_id=qa_id,
        question=answer["question"],
        answer=answer["answer"],
        sources=sources,
        date_added=(
            datetime.fromtimestamp(int(answer["date_added"]), tz=timezone.utc)
            if answer.get("date_added")
            else None
        ),
        date_added_ts=(
            humanize_unix_date(answer["date_added"]) if answer.get("date_added") else ""
        ),
    )


@app.get("/answers/{qa_id}/html", response_class=HTMLResponse)
def get_answer_markdown(
    request: Request,
    qa_id: str,
    include_metadata: bool = True,
    hx_request: Annotated[bool, Header()] = None,
    hx_boosted: Annotated[bool, Header()] = None,
):
    """
    Retrieve a specific answer by its ID and return a markdown response.

    Args:
        qa_id: The ID of the answer to retrieve.

    Returns:
        HTMLResponse: A rendered markdown string of the question and answer
        Raises HTTPException if not found.
    """

    answer = get_answer_by_id(qa_id=qa_id, include_metadata=include_metadata)
    # Convert markdown to html
    html_content = ""
    sources_str = ""
    for source in answer.sources:
        sources_str += f"""
        <li>{source.title + " |" if source.title else ""} <a href='{source.source}'>{source.source}</a>
        </li>
        """
    html_content += f"""
    <article class="container-fluid" style="margin: 2rem; width: 95%;">
        <header><h1><a href='/answers/{answer.qa_id}/html'>{answer.question}</a></h1></header>
        {mistune.html(PyMarkdownApi().fix_string(answer.answer).fixed_file)}
        <h6>Sources:</h6>
            <ul>{sources_str}</ul>
            <footer>{answer.date_added_ts}</footer>
            </article>
"""
    if hx_request and hx_boosted:
        return html_content
    else:
        return templates.TemplateResponse(
                "static/index.html", {"request": request, "html_content": html_content}
            )


@app.get("/search", response_class=HTMLResponse)
def get_search_results(
    request: Request,
    hx_request: Annotated[str | None, Header()] = None,
    q: Optional[str] = Query(default=None, description="Search for questions"),
    limit: int = Query(
        default=10, ge=1, le=100, description="Limit the number of results per page"
    ),
    page_no: int = Query(default=1, description="Which page number of results to get"),
    include_metadata: bool = Query(
        default=False, description="Include document metadata"
    ),
):
    """
    Retrieve answers from Redis based on search terms.

    Args:
        q: (Optional) Search term to match against the question.
        source: (Optional) Search term to filter by source.
        limit: (Optional) Maximum number of results to return.
        page_no: (Optional) Page number of results to return.
        include_metadata: (Optional) Flag to include metadata in the response

    Returns:
         HTMLResponse: A rendered markdown string of the question and answer
    """
    try:
        answers = get_answers(
            q=q, limit=limit, page_no=page_no, include_metadata=include_metadata
        ).results
        n_answers = len(answers)
        html_content = ''
        for i_answer, answer in enumerate(answers):
            sources_str = ""
            for source in answer.sources:
                sources_str += f"""
                <li>{source.title + "  |" if source.title else ""} <a href='{source.source}'>{source.source}</a>
                </li>
               """
            if (i_answer + 1) < n_answers:
                html_content += f"""<div class="container"><article
                >"""
            else:
                html_content += f"""
            <div class="container"><article hx-get="/search?q={q}&page_no={page_no+1}&limit={limit}&include_metadata={include_metadata}"
            hx-trigger="revealed"
    hx-swap="afterend"
    hx-indicator="#loading"
            >"""

            html_content += f"""
                <header><h2><a hx-boost='true' hx-target="#results" hx-swap="innerHTML" href='/answers/{answer.qa_id}/html?q={q}&'>{answer.question}</a></h2></header>
        {mistune.html(PyMarkdownApi().fix_string(answer.answer).fixed_file)}
        <h6>Sources:</h6>
                 <ul>{sources_str}</ul>
                 <footer>{answer.date_added_ts}</footer>
                 </article></div>
        """
        if hx_request:
            return html_content
        return templates.TemplateResponse(
            "static/index.html", {"request": request, "html_content": html_content}
        )
    except Exception as e:
        logging.error(f"Error querying Redis: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error querying Redis: {e}")


app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
