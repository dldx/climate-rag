import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Annotated, List, Optional
from urllib.parse import quote_plus, urlencode

import gradio as gr
import msgspec
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from cache import r
from helpers import get_previous_queries, humanize_unix_date
from schemas import SourceMetadata
from tools import (
    clean_urls,
    get_source_document_extra_metadata,
    get_sources_based_on_filter,
)
from webapp import demo

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

templates = Jinja2Templates(directory="./static")
templates.env.filters["urlencode"] = lambda x: urlencode(x) if x else ""


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
    if q is None or q == "" or q == "None":
        q = "*"
    elif "@" not in q:
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
    html_content = r""
    sources_str = ""
    for source in answer.sources:
        source_metadata = [
            f"<strong>{source.title}</strong>",
            source.company_name,
            source.publishing_date.strftime("%B %Y")
            if source.publishing_date
            else None,
            f"<a target='_blank' href='{source.source}'>{source.source}</a>",
        ]
        source_str = ", ".join(filter(None, source_metadata))
        sources_str += f"""<li>{source_str}</li>"""
    # To deal with double backslashes and backticks in the markdown, we need to escape them so that they are not interpreted as escape characters.
    # This is necessary to prevent latex in markdown from being interpreted incorrectly.
    answer.answer = answer.answer.replace("\\", "\\\\").replace("`", "\\`")
    html_content += f"""
    <article class="container-fluid" style="margin: 2rem; width: 95%;">
        <header><h1><a href='/answers/{quote_plus(answer.qa_id)}/html'>{answer.question}</a></h1></header>
        <div>
            <script>
            renderMarkdownWithKatex(`{answer.answer}`);
            </script>
        </div>
        <h6>Sources:</h6>
        <ul>{sources_str}</ul>
        <footer>{answer.date_added_ts}</footer>
    </article>
"""
    if hx_request and hx_boosted:
        return html_content
    else:
        return templates.TemplateResponse(
            "qa.html",
            {"request": request, "html_content": html_content},
        )


@app.get("/search_qa", response_class=HTMLResponse)
def get_answer_search_results(
    request: Request,
    hx_request: Annotated[str | None, Header()] = None,
    hx_boosted: Annotated[str | None, Header()] = None,
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
        html_content = ""
        for i_answer, answer in enumerate(answers):
            sources_str = ""
            # for source in answer.sources:
            #     source_metadata = [
            #         source.company_name,
            #         source.title,
            #         source.publishing_date.strftime("%B %Y")
            #         if source.publishing_date
            #         else None,
            #         f"<a href='{source.source}'>{source.source}</a>",
            #     ]
            #     source_str = ", ".join(filter(None, source_metadata))
            #     sources_str += f"""<li>{source_str}</li>"""
            if (i_answer + 1) < n_answers:
                html_content += """<div class="container"><article
                >"""
            else:
                next_page_params = {
                    "q": q,
                    "page_no": page_no + 1,
                    "limit": limit,
                    "include_metadata": include_metadata,
                }
                next_page_url = f"/search_qa?{urlencode(next_page_params)}"
                html_content += f"""
            <div class="container"><article hx-get="{next_page_url}"
            hx-trigger="revealed"
    hx-swap="afterend"
    hx-indicator="#loading"
            >"""

            # To deal with double backslashes and backticks in the markdown, we need to escape them so that they are not interpreted as escape characters.
            # This is necessary to prevent latex in markdown from being interpreted incorrectly.
            answer.answer = answer.answer.replace("\\", "\\\\").replace("`", "\\`")
            html_content += f"""
                <header><h2><a hx-boost='true' hx-target="#results" hx-swap="innerHTML" href='/answers/{quote_plus(answer.qa_id)}/html?q={q}&'>{answer.question}</a></h2></header>
        <div>
            <script>
            renderMarkdownWithKatex(`{answer.answer}`);
            </script>
        </div>
        <!--<<h6>Sources:</h6>
                 ul>{sources_str}</ul>-->
                 <footer>{answer.date_added_ts}</footer>
                 </article></div>
        """
        if hx_request and not hx_boosted:
            return html_content
        return templates.TemplateResponse(
            "qa.html",
            {"request": request, "html_content": (html_content)},
        )
    except Exception as e:
        logging.error(f"Error querying Redis: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error querying Redis: {e}")


class Source(BaseModel):
    source: str
    page_content: str
    date_added: datetime
    date_added_ts: Optional[str] = ""
    title: Optional[str] = None
    company_name: Optional[str] = None
    publishing_date: Optional[datetime] = None


class Sources(BaseModel):
    results: List[Source]


@app.get("/sources", response_model=Sources)
def get_sources(
    q: Optional[str] = Query(default=None, description="Search for sources"),
    limit: int = Query(
        default=5, ge=1, le=1000, description="Limit the number of results"
    ),
    page_no: int = Query(default=1, description="Which page number of results to get"),
):
    if q is None:
        q = ""
    if "@" not in q:
        q = f"@source:({q})"  # or @title, etc

    sources = get_sources_based_on_filter(q, r, limit=limit, page_no=page_no)

    results = []
    for source_url in sources:
        source_data = r.hgetall(f"climate-rag::source:{source_url}")
        if source_data:  # Check if source_data exists
            results.append(
                Source(
                    source=source_url,
                    page_content=source_data["page_content"],
                    date_added=(
                        datetime.fromtimestamp(
                            int(source_data["date_added"]), tz=timezone.utc
                        )
                        if source_data.get("date_added")
                        else None
                    ),
                    date_added_ts=(
                        humanize_unix_date(source_data["date_added"])
                        if source_data.get("date_added")
                        else ""
                    ),
                    title=source_data.get("title"),
                    company_name=source_data.get("company_name"),
                    publishing_date=(
                        datetime.fromtimestamp(
                            int(source_data["publishing_date"]), tz=timezone.utc
                        )
                        if source_data.get("publishing_date")
                        else None
                    ),
                )
            )

    # results = sorted(results, key=lambda x: x.date_added, reverse=True)

    return Sources(results=results)


@app.get("/search_sources", response_class=HTMLResponse)
def get_source_search_results(
    request: Request,
    hx_request: Annotated[str | None, Header()] = None,
    hx_boosted: Annotated[str | None, Header()] = None,
    q: Optional[str] = Query(default=None, description="Search for sources"),
    limit: int = Query(
        default=5, ge=1, le=1000, description="Limit the number of results per page"
    ),
    page_no: int = Query(default=1, description="Which page number of results to get"),
):
    """
    Retrieve sources from Redis based on search terms.

    Args:
        q: (Optional) Search term to match against the source.
        limit: (Optional) Maximum number of results to return.
        page_no: (Optional) Page number of results to return.

    Returns:
         HTMLResponse: A rendered HTML string of the sources
    """
    try:
        sources_results = get_sources(q=q, limit=limit, page_no=page_no).results
        n_sources = len(sources_results)

        html_content = ""
        for i_source, source in enumerate(sources_results):
            if (i_source + 1) < n_sources:
                html_content += "<div class='container'><article>"
            else:
                # HTMX magic for infinite scrolling
                next_page_params = {
                    "q": q,
                    "page_no": page_no + 1,
                    "limit": limit,
                }
                next_page_url = f"/search_sources?{urlencode(next_page_params)}"
                html_content += f"""<div class="container">
                <article hx-get="{next_page_url}"
                        hx-trigger="revealed"
                        hx-swap="afterend"
                        hx-indicator="#loading">"""

            publishing_date_str = (
                source.publishing_date.strftime("%Y-%m-%d")
                if source.publishing_date
                else "Unknown"
            )
            # try:
            #     source_md = PyMarkdownApi().fix_string(source.page_content).fixed_file
            # except:
            source_md = source.page_content
            html_content += f"""
                <header><h2><a target='_blank' href='{source.source}'>{source.title or source.source}</a></h2></header>
                <p>Company: {source.company_name or "Unknown"}</p>
                <p>Publishing Date: {publishing_date_str}</p>
                <p>Date Added: {source.date_added_ts}</p>
                <pre style="max-height: 100ch; white-space: pre-wrap;">
            <script>
                document.currentScript.parentElement.innerHTML = DOMPurify.sanitize(marked.parse(`{source_md if source.page_content else ""}`));
            </script></pre>  </article></div>
            """

        if hx_request and not hx_boosted:
            return html_content

        return templates.TemplateResponse(
            "sources.html", {"request": request, "html_content": html_content}
        )

    except Exception as e:
        logging.error(f"Error querying Redis: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error querying Redis: {e}")


@app.get("/sources/{source}", response_model=Source)
def get_source_by_id(source: str):
    """
    Retrieve a specific source by its ID.

    Args:
        source: The ID of the source to retrieve.

    Returns:
        dict: A dictionary representing the matching source, if found.
        Raises HTTPException if not found.
    """
    source_data = r.hgetall(f"climate-rag::source:{source}")
    if not source_data:
        raise HTTPException(status_code=404, detail="Source not found")

    return Source(
        source=source,
        page_content=source_data["page_content"],
        date_added=(
            datetime.fromtimestamp(int(source_data["date_added"]), tz=timezone.utc)
            if source_data.get("date_added")
            else None
        ),
        date_added_ts=(
            humanize_unix_date(source_data["date_added"])
            if source_data.get("date_added")
            else ""
        ),
        title=source_data.get("title"),
        company_name=source_data.get("company_name"),
        publishing_date=(
            datetime.fromtimestamp(int(source_data["publishing_date"]), tz=timezone.utc)
            if source_data.get("publishing_date")
            else None
        ),
    )


app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
