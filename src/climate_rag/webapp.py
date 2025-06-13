import os
import re
import sys
from typing import List, Tuple

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from gradio_log import Log
from ulid import ULID

from climate_rag.cache import r, source_index_name
from climate_rag.constants import language_choices
from climate_rag.helpers import (
    clean_urls,
    compile_answer,
    render_qa_pdfs,
    sanitize_url,
)
from climate_rag.logo import climate_rag_logo
from climate_rag.query_data import query_source_documents, run_query
from climate_rag.tools import get_vector_store

sys.settrace(None)

load_dotenv()
# Initialize with default project
default_project_id = "langchain"
db = get_vector_store(default_project_id)


def download_latest_answer(
    questions: List[str], answers: List[str]
) -> Tuple[gr.DownloadButton, gr.DownloadButton]:
    """
    Returns the download buttons for the latest answer as PDF or DOCX.

    Args:
        questions (List[str]): The list of questions.
        answers (List[str]): The list of answers.
    """

    if (len(answers) == 0) or (len(questions) > len(answers)):
        return gr.DownloadButton(visible=False), gr.DownloadButton(visible=False)
    qa_id = answers[-1]

    pdf_download_url, docx_download_url = render_qa_pdfs(qa_id)

    return gr.DownloadButton(
        value=sanitize_url(docx_download_url), visible=True
    ), gr.DownloadButton(value=sanitize_url(pdf_download_url), visible=True)


# Project management functions
def get_projects():
    """Get a list of available projects from Redis"""
    try:
        projects = r.smembers("climate-rag::projects")
        if not projects:
            # Initialize with default project if no projects exist
            r.sadd("climate-rag::projects", "langchain")
            projects = ["langchain"]
        else:
            projects = [p.decode() if isinstance(p, bytes) else p for p in projects]

        # Sort projects with default always first
        projects = sorted([p for p in projects if p != "langchain"])
        projects.insert(0, "langchain")
        return projects
    except Exception as e:
        print(f"Error getting projects: {e}")
        return ["langchain"]


def refresh_projects():
    """Refresh the list of available projects"""
    projects = get_projects()
    return gr.Dropdown(choices=projects, value=projects[0])


def create_project(project_name):
    """Create a new project"""
    if not project_name or len(project_name.strip()) == 0:
        return "Please enter a valid project name", gr.Dropdown()

    # Clean project name (remove special characters, spaces, etc.)
    project_name = re.sub(r"[^a-zA-Z0-9_-]", "_", project_name.strip().lower())

    # Check if project already exists
    if project_name in get_projects():
        return f"Project '{project_name}' already exists", gr.Dropdown()

    # Initialize project indices
    from climate_rag.tools import initialize_project_indices

    if initialize_project_indices(r, project_name):
        # Refresh dropdown
        projects = get_projects()
        return f"Project '{project_name}' created successfully", gr.Dropdown(
            choices=projects, value=project_name
        )
    else:
        return f"Error creating project '{project_name}'", gr.Dropdown()


# Helper function to count documents in a project
def count_project_documents(project_id):
    """Count all documents in a project"""

    try:
        return r.ft(f"{source_index_name}_{project_id}").info()["num_docs"]
    except Exception as e:
        print(f"Error counting documents in project {project_id}: {e}")
        return 0


def list_project_documents(project_id):
    """List all documents in a project"""
    from climate_rag.query_data import query_source_documents

    # Get all documents from the project
    db = get_vector_store(project_id)
    results = query_source_documents(
        db,
        "*",
        print_output=False,
        fields=["title", "company_name", "source", "date_added", "page_length"],
        limit=1000,
        project_id=project_id,
    )

    if len(results) == 0:
        return gr.DataFrame(
            value=pd.DataFrame(
                columns=["Title", "Company", "Source", "Date Added", "Length"]
            )
        )

    # Create DataFrame with the results
    df = results[["title", "company_name", "source", "date_added", "page_length"]]
    df.columns = ["Title", "Company", "Source", "Date Added", "Length"]

    return gr.DataFrame(value=df)


def move_document(source_uri, source_project, target_project):
    """Move a document from one project to another"""
    from climate_rag.tools import transfer_document_between_projects

    if source_project == target_project:
        return f"Source and target projects are the same: {source_project}"

    source_db = get_vector_store(source_project)
    success = transfer_document_between_projects(
        source_uri, source_project, target_project, r, source_db, delete_source=True
    )

    if success:
        return f"Document {source_uri} moved from {source_project} to {target_project}"
    else:
        return f"Failed to move document {source_uri}"


def copy_document(source_uri, source_project, target_project):
    """Copy a document from one project to another"""
    from climate_rag.tools import transfer_document_between_projects

    if source_project == target_project:
        return f"Source and target projects are the same: {source_project}"

    source_db = get_vector_store(source_project)
    success = transfer_document_between_projects(
        source_uri, source_project, target_project, r, source_db, delete_source=False
    )

    if success:
        return f"Document {source_uri} copied from {source_project} to {target_project}"
    else:
        return f"Failed to copy document {source_uri}"


def delete_project(project_id):
    """Delete a project and all its documents"""
    # Don't allow deletion of the default project
    if project_id == "langchain":
        return "Cannot delete the default project"

    try:
        # Delete project from Redis set
        r.srem("climate-rag::projects", project_id)

        # Delete indices for the project
        project_source_index_name = f"idx:source_{project_id}"
        project_zh_source_index_name = f"idx:source_zh_{project_id}"
        project_ja_source_index_name = f"idx:source_ja_{project_id}"

        try:
            r.ft(project_source_index_name).dropindex(delete_documents=True)
        except Exception as e:
            print(f"Error dropping index {project_source_index_name}: {e}")

        try:
            r.ft(project_zh_source_index_name).dropindex(delete_documents=True)
        except Exception as e:
            print(f"Error dropping index {project_zh_source_index_name}: {e}")

        try:
            r.ft(project_ja_source_index_name).dropindex(delete_documents=True)
        except Exception as e:
            print(f"Error dropping index {project_ja_source_index_name}: {e}")

        # Delete the Chroma collection for the project
        import chromadb

        client = chromadb.HttpClient(
            host=os.environ["CHROMADB_HOSTNAME"],
            port=int(os.environ["CHROMADB_PORT"]),
        )

        try:
            client.delete_collection(project_id)
        except Exception as e:
            print(f"Error deleting project: {e}")
            pass

        return f"Project '{project_id}' deleted successfully"
    except Exception as e:
        return f"Error deleting project '{project_id}': {str(e)}"


def climate_chat(
    message,
    history,
    questions,
    answers,
    chat_id,
    rag_filter,
    improve_question,
    do_rerank,
    do_crawl,
    max_search_queries,
    do_add_additional_metadata,
    language,
    initial_generation,
    project_id,
):
    happy_with_answer = True
    getting_feedback = False
    USER_FEEDBACK_QUESTION = """Are you happy with the answer? (y / n=web search)"""
    answer = ""
    number_of_past_questions = 0
    if number_of_past_questions != len(questions):
        getting_feedback = False
        happy_with_answer = True
    if len(history) > 1:
        getting_feedback = history[-2][1] == USER_FEEDBACK_QUESTION
        if getting_feedback:
            if message[0].lower() == "n":
                happy_with_answer = False
                yield None, questions, answers
            elif message[0].lower() == "y":
                happy_with_answer = True
                getting_feedback = False
                yield (
                    "Great! I'm glad I could help. What else would you like to know?",
                    questions,
                    answers,
                )
                number_of_past_questions = len(questions)
                return
            else:
                happy_with_answer = True
                getting_feedback = False
                number_of_past_questions = len(questions)
                yield None, questions, answers

    if getting_feedback:
        message = questions[-1]
    else:
        questions.append(message)
    if rag_filter == "":
        rag_filter = None
    else:
        if bool(re.search(r"@.+:.+", rag_filter)) is False:
            # Turn it into a valid filter
            rag_filter = f"@source:*{rag_filter}*"

    for key, value in run_query(
        message,
        llm="gemini-2.5-flash-preview-05-20",
        mode="gui",
        rag_filter=rag_filter,
        improve_question=improve_question,
        language=language,
        do_rerank=do_rerank,
        do_crawl=do_crawl,
        max_search_queries=max_search_queries,
        do_add_additional_metadata=do_add_additional_metadata,
        history=history,
        initial_generation=initial_generation,
        happy_with_answer=happy_with_answer,
        continue_after_interrupt=getting_feedback,
        thread_id=chat_id,
        project_id=project_id,
    ):
        if key == "improve_question":
            if improve_question:
                yield (
                    f"""**Improved question:** {value["question"]}"""
                    + (
                        f"""\n**Better question (en):** {value["question_en"]}"""
                        if language != "en"
                        else ""
                    ),
                    questions,
                    answers,
                )
            else:
                yield None, questions, answers
        elif key == "formulate_query":
            yield (
                """**Generated search queries:**\n\n"""
                + "\n".join(
                    [
                        (
                            f" * {query.query} ({query.query_en})"
                            if language != "en"
                            else f" * {query.query_en}"
                        )
                        for query in value["search_prompts"]
                    ]
                ),
                questions,
                answers,
            )

        elif key == "retrieve_from_database":
            yield "**Retrieving documents from database...**", questions, answers
            # yield f"""Search queries: {value["search_prompts"]}

            # Retrieved from database: {[doc.metadata["id"] for doc in value["documents"]]}""", questions
        elif key == "web_search_node":
            yield (
                """**Searching the web for more information...**""",
                questions,
                answers,
            )
            # yield f"""Search query: {value["search_query"]}

            # Search query (en): {value["search_query_en"]}""", questions
        elif key == "add_additional_metadata":
            yield (
                """**Fetching document titles, entities, etc...**""",
                questions,
                answers,
            )
        elif key == "rerank_documents":
            # yield f"""Reranked documents: {[doc.metadata["id"] for doc in value["documents"]]}""", questions
            yield """**Reranking documents...**""", questions, answers

        elif key == "generate":
            answers.append(value["qa_id"])
            answer = compile_answer(
                value["generation"],
                value["initial_question"],
                [doc.metadata.get("source") for doc in value["documents"]],
            )

            yield answer, questions, answers
            # After generation ask for feedback
            yield USER_FEEDBACK_QUESTION, questions, answers
        elif key == "add_urls_to_database":
            yield """**Added new pages to database**""", questions, answers
        # else:
        #     yield str(value), questions, answers


with gr.Blocks(
    theme=gr.themes.Monochrome(),
    title="Climate RAG",
    fill_height=True,
    css="""
# .h-full {
#     height: 85vh;
# }
# .h-20 {
#     height: 20px;
# }
.scroll-y {
 overflow-y: scroll;
}

.tabs {
    top: -30px;
}


.gradio-container {
    padding: 0 !important;
}

.upload-button {
    display: none !important;

}

footer {
    display:none !important
}

""",
) as demo:
    # Define how to store state
    chat_state = gr.State([])
    questions_state = gr.State([])
    answers_state = gr.State([])
    chat_id_state = gr.State("")

    gr.HTML(climate_rag_logo)

    with gr.Tab("Chat"):
        # Add a header
        with gr.Row(elem_classes=["h-full"]):
            with gr.Column(variant="panel"):
                # Add project selector dropdown
                with gr.Column(scale=1, variant="panel"):
                    project_dropdown = gr.Dropdown(
                        choices=get_projects(),  # Get actual projects
                        value="langchain",
                        label="Select Project",
                        info="Choose which project to query",
                        interactive=True,
                    )

                    # Add a button to refresh projects list
                    refresh_projects_button = gr.Button("Refresh projects")
                improve_question_checkbox = gr.Checkbox(
                    value=True, label="Auto-improve question?"
                )
                do_initial_generation_checkbox = gr.Checkbox(
                    value=True, label="Generate answer before web search?"
                )
                do_rerank_checkbox = gr.Checkbox(value=True, label="Rerank documents?")
                do_crawl_checkbox = gr.Checkbox(
                    value=False, label="Crawl within search results?"
                )
                do_add_additional_metadata_checkbox = gr.Checkbox(
                    value=True, label="Augment with additional metadata?"
                )
                max_search_queries_textbox = gr.Number(
                    label="Maximum number of search queries to run",
                    value=1,
                    minimum=1,
                    maximum=15,
                )
                language_dropdown = gr.Dropdown(
                    choices=language_choices,
                    label="Language",
                    value="en",
                    allow_custom_value=True,
                    filterable=True,
                )
                rag_filter_textbox = gr.Textbox(
                    placeholder="eg. carbontracker or shell*annual",
                    label="Database filter",
                )
                doc_sources = gr.Dataset(
                    components=[
                        gr.Textbox(visible=False),
                    ],
                    label="Filtered documents",
                    samples_per_page=5,
                    headers=["Source"],
                )

                # Connect project management buttons
                refresh_projects_button.click(
                    fn=refresh_projects, inputs=[], outputs=[project_dropdown]
                )

            with gr.Column(scale=4):
                chat_header = gr.Markdown("## Chat", height=100)
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    type="tuples",
                    scale=4,
                    show_label=False,
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "\\[", "right": "\\]", "display": True},
                    ],
                )
                with gr.Row():
                    chat_input = gr.MultimodalTextbox(
                        elem_id="msg-box",
                        interactive=True,
                        placeholder="Ask a question. e.g. What does the Shell 2023 annual report say about climate change?",
                        max_lines=4,
                        file_types=[],
                        show_label=False,
                        scale=6,
                    )
                with gr.Row():
                    stop_button = gr.Button(value="Stop", variant="stop", scale=5)
                    # download_word_button = gr.DownloadButton(
                    #     icon="static/ri--file-word-line.svg",
                    #     label="Download DOCX",
                    #     size="sm",
                    #     scale=1,
                    #     visible=False,
                    # )
                    download_pdf_button = gr.DownloadButton(
                        icon="static/ri--file-pdf-line.svg",
                        label="Download PDF",
                        size="sm",
                        scale=1,
                        visible=False,
                    )

    with gr.Tab(
        "Previous queries", elem_classes=["h-full", "scroll-y"]
    ) as previous_queries_tab:
        with gr.Row():
            query_history_search = gr.Textbox(
                placeholder="Search previous queries", show_label=False
            )
        with gr.Row():
            query_history_display = gr.Dataset(
                components=[
                    gr.Textbox(visible=False),
                    gr.Markdown(
                        visible=False,
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "\\[", "right": "\\]", "display": True},
                        ],
                    ),
                    gr.Markdown(
                        visible=False,
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "\\[", "right": "\\]", "display": True},
                        ],
                        elem_classes=["scroll-y", "h-20"],
                    ),
                    gr.HTML(visible=False),
                    gr.HTML(visible=False),
                ],
                label="History",
                headers=["Date", "Question", "Answer", "PDF"],
            )

    with gr.Tab("Documents", elem_classes=["h-full"]):
        with gr.Row():
            gr.Markdown("## Add new documents")
        with gr.Row():
            with gr.Column(scale=1):
                new_file = gr.File(
                    label="Upload documents",
                    file_count="multiple",
                    type="filepath",
                )
            with gr.Column(scale=4):
                url_input = gr.Textbox(placeholder="Enter a URL", show_label=False)
                add_button = gr.Button(value="Add/View")
                selected_source = gr.Markdown(height=200)
        with gr.Row():
            gr.Markdown("## Search through existing documents")
        # Search through documents in the vector database
        with gr.Row():
            search_input = gr.Textbox(
                placeholder='Search documents. eg. carbontracker or @title:"annual report 2022 shell"',
                show_label=False,
            )
            search_button = gr.Button(value="Search", visible=False)
        with gr.Row():
            search_results_display = gr.Dataset(
                components=[
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Number(visible=False),
                ],
                label="Documents retrieved",
                headers=[
                    "Title",
                    "Company name",
                    "Source",
                    "Date added",
                    "Page length",
                ],
            )
    with gr.Tab("Console"):
        log_file = "rag.log"
        Log(
            log_file=log_file,
            dark=True,
            tail=1000,
            elem_classes=["h-full"],
            min_width=1000,
        )

    with gr.Tab("Project Management"):
        with gr.Row():
            gr.Markdown("## Project Management")

        with gr.Row():
            with gr.Column(scale=1):
                # New project creation section
                new_project_name = gr.Textbox(
                    label="New Project Name",
                    info="Enter a name for a new project",
                    placeholder="Enter project name...",
                    interactive=True,
                )
                create_project_button = gr.Button("Create New Project")
                create_project_status = gr.Textbox(
                    label="Creation Status", interactive=False
                )

            with gr.Column(scale=1):
                project_list = gr.Dropdown(
                    choices=get_projects(),
                    label="Select Project to Manage",
                    info="Choose a project to view or modify",
                    interactive=True,
                )
                refresh_project_list_button = gr.Button("Refresh Projects")
                project_docs_count = gr.Number(
                    label="Document Count", interactive=False
                )
                delete_project_button = gr.Button("Delete Project", variant="stop")
                delete_project_status = gr.Textbox(
                    label="Deletion Status", interactive=False
                )

        # Connect project creation event handler
        create_project_button.click(
            fn=create_project,
            inputs=[new_project_name],
            outputs=[create_project_status],
        ).then(
            fn=refresh_projects,
            inputs=[],
            outputs=[project_list],
        ).then(
            fn=lambda: "",  # Clear the input field after creation
            inputs=[],
            outputs=[new_project_name],
        )

        with gr.Row():
            gr.Markdown("## Document Operations")

        with gr.Row():
            with gr.Column(scale=1):
                source_project = gr.Dropdown(
                    choices=get_projects(),
                    label="Source Project",
                    info="Project containing the document",
                    interactive=True,
                )
                source_document = gr.Textbox(
                    label="Document URI",
                    info="URI of the document to move/copy",
                    interactive=True,
                )

            with gr.Column(scale=1):
                target_project = gr.Dropdown(
                    choices=get_projects(),
                    label="Target Project",
                    info="Project to move/copy the document to",
                    interactive=True,
                )
                operation_result = gr.Textbox(
                    label="Operation Result", interactive=False
                )

        with gr.Row():
            move_doc_button = gr.Button("Move Document")
            copy_doc_button = gr.Button("Copy Document")

    ### Define the logic
    ## Tab 1: Chat

    def user(user_message, history):
        if (len(user_message["text"]) > 10) or (user_message["text"]) in ["y", "n"]:
            return (
                {"text": "", "files": []},
                history + [[user_message["text"], None]],
                history + [[user_message["text"], None]],
            )
        else:
            return (
                {"text": user_message["text"], "files": []},
                history,
                history,
            )

    def bot(
        chat_history,
        questions,
        answers,
        chat_id,
        rag_filter,
        improve_question,
        do_rerank,
        do_crawl,
        max_search_queries,
        do_add_additional_metadata,
        language,
        initial_generation,
        project_id,
    ):
        if len(chat_history) == 0:
            chat_history.append(
                [
                    None,
                    "Hello! I'm here to help you with climate-related questions. What would you like to know?",
                ]
            )
            return chat_history, chat_history, questions, None
        message = chat_history[-1][0]
        chat_id = chat_id or str(ULID())
        bot_messages = climate_chat(
            message=message,
            history=chat_history,
            questions=questions,
            answers=answers,
            chat_id=chat_id,
            rag_filter=rag_filter,
            improve_question=improve_question,
            do_rerank=do_rerank,
            do_crawl=do_crawl,
            max_search_queries=max_search_queries,
            do_add_additional_metadata=do_add_additional_metadata,
            language=language,
            initial_generation=initial_generation,
            project_id=project_id,
        )
        for bot_message, questions, answers in bot_messages:
            chat_history.append([None, bot_message])
            yield (
                chat_history,
                chat_history,
                questions,
                answers,
                chat_id,
                download_latest_answer(questions, answers)[1],
                f"## Chat\n### {questions[-1]}",
            )

    converse_event = chat_input.submit(
        fn=user,
        inputs=[chat_input, chat_state],
        outputs=[chat_input, chatbot, chat_state],
        queue=False,
    ).then(
        bot,
        [
            chatbot,
            questions_state,
            answers_state,
            chat_id_state,
            rag_filter_textbox,
            improve_question_checkbox,
            do_rerank_checkbox,
            do_crawl_checkbox,
            max_search_queries_textbox,
            do_add_additional_metadata_checkbox,
            language_dropdown,
            do_initial_generation_checkbox,
            project_dropdown,
        ],
        [
            chatbot,
            chat_state,
            questions_state,
            answers_state,
            chat_id_state,
            # download_word_button,
            download_pdf_button,
            chat_header,
        ],
    )

    def stop_querying(questions):
        # Remove the last question
        if len(questions) > 0:
            questions.pop()
        return questions, "## Chat"

    stop_button.click(
        fn=stop_querying,
        inputs=[questions_state],
        outputs=[questions_state, chat_header],
        cancels=[converse_event],
        queue=False,
    )

    def filter_documents(search_query, project_id="langchain"):
        from climate_rag.query_data import query_source_documents

        if search_query is None:
            search_query = ""
        if len(search_query) < 2:
            return gr.Dataset(samples=[])
        if bool(re.search(r"@.+:.+", search_query)) is False:
            # Turn into a search query if not a ft.search-style query
            search_query = f"@source:*{search_query}*"

        search_results = query_source_documents(
            db,
            search_query,
            print_output=False,
            fields=["source"],
            limit=30,
            project_id=project_id,
        )[["source"]].iloc[:30]
        search_results["source"] = clean_urls(search_results["source"].tolist())
        # Remove static path to simplify display
        search_results["source"] = search_results["source"].apply(
            lambda x: (
                ("🗎 " + x.split("/")[-1])
                if os.environ.get("STATIC_PATH", "") in str(x)
                else x
            )
        )

        return gr.Dataset(samples=search_results.to_numpy().tolist())

    rag_filter_textbox.change(
        fn=lambda x, y: filter_documents(x, y),
        inputs=[rag_filter_textbox, project_dropdown],
        outputs=[doc_sources],
        queue=False,
    )

    ## Tab 2: Query history
    def get_query_history(query_filter):
        from climate_rag.helpers import get_previous_queries

        if (query_filter is None) or (query_filter == ""):
            query_filter = "*"

        # Get dataframe of previous queries
        df = get_previous_queries(r, query_filter=query_filter, limit=100).set_index(
            "qa_id"
        )[["date_added_ts", "question", "answer", "pdf_uri"]]
        df.question = (
            '<a target="_blank" href="/answers/'
            + df.index
            + '/html" >'
            + df.question
            + "</a>"
        )
        # Deal with missing PDF and DOCX URIs
        missing_pdfs = df.loc[lambda df: df.isnull().any(axis=1)].index.tolist()
        for qa_id in missing_pdfs:
            df.loc[qa_id, "pdf_uri"] = render_qa_pdfs(qa_id)[0]
        # Sanitize URLs
        df.pdf_uri = df.pdf_uri.apply(sanitize_url)
        # df.docx_uri = df.docx_uri.apply(sanitize_url)

        df.pdf_uri = (
            "<a target='_blank' href='"
            + df.pdf_uri.astype(str).fillna("")
            + "'><img src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDI0IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPHBhdGggZD0iTTUgNGgxMHY0aDR2MTJINXpNMy45OTkgMkEuOTk1Ljk5NSAwIDAgMCAzIDIuOTkydjE4LjAxNmExIDEgMCAwIDAgLjk5My45OTJoMTYuMDE0QTEgMSAwIDAgMCAyMSAyMC45OTJWN2wtNS01em02LjUgNS41YzAgMS41NzctLjQ1NSAzLjQzNy0xLjIyNCA1LjE1M2MtLjc3MiAxLjcyMy0xLjgxNCAzLjE5Ny0yLjkgNC4wNjZsMS4xOCAxLjYxM2MyLjkyNy0xLjk1MiA2LjE2OC0zLjI5IDkuMzA0LTIuODQybC40NTctMS45MzlDMTQuNjQ0IDEyLjY2MSAxMi41IDkuOTkgMTIuNSA3LjV6bS42IDUuOTcyYy4yNjgtLjU5Ny41MDUtMS4yMTYuNzA1LTEuODQzYTkuNyA5LjcgMCAwIDAgMS43MDYgMS45NjZjLS45ODIuMTc2LTEuOTQ0LjQ2NS0yLjg3NS44MzNxLjI0OC0uNDcxLjQ2NS0uOTU2IiBmaWxsPSIjZmZmIiBzdHJva2U9IiMwMDAiIHN0cm9rZS13aWR0aD0iLjIiLz4KPC9zdmc+Cg==' alt='PDF'></img></a>"
        )
        # df.docx_uri = (
        #     "<a target='_blank' href='"
        #     + df.docx_uri.astype(str).fillna("")
        #     + "'><img src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDI0IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPHBhdGggZD0iTTE2IDh2OGgtMmwtMi0ybC0yIDJIOFY4aDJ2NWwyLTJsMiAyVjhoMVY0SDV2MTZoMTRWOHpNMyAyLjk5MkMzIDIuNDQ0IDMuNDQ3IDIgMy45OTkgMkgxNmw1IDV2MTMuOTkzQTEgMSAwIDAgMSAyMC4wMDcgMjJIMy45OTNBMSAxIDAgMCAxIDMgMjEuMDA4eiIgZmlsbD0iI2ZmZiIgc3Ryb2tlPSIjMDAwIiBzdHJva2Utd2lkdGg9Ii4yIi8+Cjwvc3ZnPg==' alt='DOCX'></img></a>"
        # )

        return gr.Dataset(samples=df.to_numpy().tolist())

    previous_queries_tab.select(
        get_query_history,
        inputs=[query_history_search],
        outputs=[query_history_display],
        queue=False,
    )
    query_history_search.change(
        fn=lambda x: get_query_history(x),
        inputs=[query_history_search],
        outputs=[query_history_display],
        queue=False,
    )

    ## Tab 3: Documents
    # Search documents
    def search_documents(search_query, project_id="langchain"):
        from climate_rag.query_data import query_source_documents

        if search_query is None:
            search_query = ""
        if len(search_query) < 2:
            return gr.Dataset(samples=[])
        if bool(re.search(r"@.+:.+", search_query)) is False:
            # Add asterisks to search query if not a ft.search-style query
            search_query = f"@source:*{search_query}*"

        search_results = query_source_documents(
            db,
            search_query,
            print_output=False,
            fields=[
                "title",
                "company_name",
                "source",
                "date_added",
                "page_length",
            ],
            limit=100,
            project_id=project_id,
        )[
            [
                "title",
                "company_name",
                "source",
                "date_added",
                "page_length",
            ]
        ]
        return gr.Dataset(samples=search_results.to_numpy().tolist())

    search_input.change(
        fn=lambda x, y: search_documents(x, y),
        inputs=[search_input, project_dropdown],
        outputs=[search_results_display],
        queue=False,
    )
    search_button.click(
        fn=lambda x, y: search_documents(x, y),
        inputs=[search_input, project_dropdown],
        outputs=[search_results_display],
        queue=False,
    )

    # Add new documents
    def add_document(url, project_id):
        from climate_rag.tools import add_urls_to_db

        add_urls_to_db([url], db, project_id=project_id)

        # Retrieve source markdown
        page_content = query_source_documents(
            db, f'@source:"{url}"', print_output=False, fields=["page_content"]
        )["page_content"]
        if len(page_content) > 0:
            page_content = page_content.iloc[0]
        else:
            page_content = f"Error loading document: {url}. See console for details."
        return page_content

    add_button.click(
        fn=add_document,
        inputs=[url_input, project_dropdown],
        outputs=[selected_source],
        queue=False,
    )
    url_input.submit(
        fn=add_document,
        inputs=[url_input, project_dropdown],
        outputs=[selected_source],
        queue=False,
    )

    # Upload new document
    from climate_rag.tools import upload_documents

    new_file.upload(
        fn=lambda x: upload_documents(x, db)[-1].metadata[
            "source"
        ],  # Return the last document added only
        inputs=[new_file],
        outputs=[search_input],
        queue=False,
    ).then(
        search_documents,
        [search_input],
        [search_results_display],
    ).then(
        lambda: None,
        inputs=None,
        outputs=[new_file],
        queue=False,
    )

    # Connect buttons to functions
    project_list.change(
        fn=count_project_documents,
        inputs=[project_list],
        outputs=[project_docs_count],
    )

    refresh_projects_button.click(
        fn=refresh_projects,
        inputs=[],
        outputs=[project_dropdown],
    )

    create_project_button.click(
        fn=create_project,
        inputs=[new_project_name],
        outputs=[create_project_status, project_dropdown],
    ).then(
        fn=refresh_projects,
        inputs=[],
        outputs=[project_dropdown],
    ).then(
        fn=lambda: "",  # Clear the input field after creation
        inputs=[],
        outputs=[new_project_name],
    )

    # Connect event handlers
    refresh_project_list_button.click(
        fn=refresh_projects,
        inputs=[],
        outputs=[project_list],
    )

    # Move document button
    move_doc_button.click(
        fn=move_document,
        inputs=[source_document, source_project, target_project],
        outputs=[operation_result],
    ).then(
        fn=count_project_documents,
        inputs=[project_list],
        outputs=[project_docs_count],
    )

    # Copy document button
    copy_doc_button.click(
        fn=copy_document,
        inputs=[source_document, source_project, target_project],
        outputs=[operation_result],
    ).then(
        fn=count_project_documents,
        inputs=[project_list],
        outputs=[project_docs_count],
    )

demo.queue(default_concurrency_limit=None)
if __name__ == "__main__":
    demo.launch(inbrowser=False, show_api=False, max_threads=80)
