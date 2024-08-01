import logging
import urllib.parse
from pathvalidate import sanitize_filename
import humanize
import msgspec
from typing import List, Tuple
import gradio as gr
from dotenv import load_dotenv
import os
import shutil
import hashlib
from langchain.schema import Document
from gradio_log import Log

load_dotenv()

from helpers import clean_urls, generate_qa_id, sanitize_url
from query_data import query_source_documents, run_query
from tools import get_vector_store, upload_file
from cache import r

db = get_vector_store()


def compile_answer(generation: str, initial_question: str, sources: List[str]) -> str:
    """
    Compile the answer from the generation and the sources.

    Args:
        generation (str): The generated answer.
        documents (List[GraphState]): The list of documents.

    Returns:
        str: The compiled answer.
    """
    answer = (
        f"""# {initial_question}\n\n"""
        + generation
        + "\n\n**Sources:**\n\n"
        + "\n\n".join(
            set(
                [
                    (" * " + clean_urls([source], os.environ.get("STATIC_PATH", ""))[0])
                    for source in sources
                    if source is not None
                ]
            )
        )
    )

    return answer


def download_latest_answer(
    questions: List[str], answers: List[str]
) -> Tuple[gr.DownloadButton, gr.DownloadButton]:
    """
    Returns the download buttons for the latest answer as PDF or DOCX.

    Args:
        questions (List[str]): The list of questions.
        answers (List[str]): The list of answers.
    """

    from helpers import md_to_pdf, pdf_to_docx

    if len(answers) == 0 or len(questions) > len(answers):
        return gr.DownloadButton(visible=False), gr.DownloadButton(visible=False)
    question = sanitize_filename(questions[-1])
    qa_id = answers[-1]

    filename = qa_id
    qa_map = r.hgetall(f"climate-rag::answer:{qa_id}")
    # Check if PDF is already in redis cache
    if qa_map.get("pdf_uri", None) is not None:
        pdf_download_url = qa_map["pdf_uri"]
        docx_download_url = qa_map["docx_uri"]
    else:
        answer = compile_answer(
            qa_map["answer"], qa_map["question"], msgspec.json.decode(qa_map["sources"])
        )
        os.makedirs("tmp", exist_ok=True)
        pdf_path = f"tmp/{filename}.pdf"
        docx_path = f"tmp/{filename}.docx"

        md_to_pdf(answer, pdf_path)
        pdf_to_docx(pdf_path, docx_path)

        STATIC_PATH = os.environ.get("STATIC_PATH", "")
        UPLOAD_FILE_PATH = os.environ.get("UPLOAD_FILE_PATH", "")
        USE_S3 = os.environ.get("USE_S3", False) == "True"

        if (STATIC_PATH != "") and (UPLOAD_FILE_PATH != ""):
            # Copy the files to the static path
            os.makedirs(f"{UPLOAD_FILE_PATH}/outputs", exist_ok=True)
            shutil.copy(pdf_path, f"{UPLOAD_FILE_PATH}/outputs/{filename}.pdf")
            shutil.copy(docx_path, f"{UPLOAD_FILE_PATH}/outputs/{filename}.docx")
            # Serve the files from the static path instead
            pdf_download_url = f"{STATIC_PATH}/outputs/{filename}.pdf"
            docx_download_url = f"{STATIC_PATH}/outputs/{filename}.docx"
        elif (STATIC_PATH != "") and (USE_S3 == True):
            # Upload the files to S3
            if not upload_file(
                file_name=pdf_path,
                bucket=os.environ.get("S3_BUCKET", ""),
                path="/outputs/",
                object_name=f"{filename}.pdf",
            ):
                logging.error(f"Failed to upload {pdf_path} to S3")
            if not upload_file(
                file_name=docx_path,
                bucket=os.environ.get("S3_BUCKET", ""),
                path="/outputs/",
                object_name=f"{filename}.docx",
            ):
                logging.error(f"Failed to upload {docx_path} to S3")
            # Serve the files from S3
            pdf_download_url = f"{STATIC_PATH}/outputs/{filename}.pdf"
            docx_download_url = f"{STATIC_PATH}/outputs/{filename}.docx"
        else:
            pdf_download_url = pdf_path
            docx_download_url = docx_path

        # Save PDF and DOCX locations to redis cache
        r.hset("climate-rag::answer:" + qa_id, "pdf_uri", pdf_download_url)
        r.hset("climate-rag::answer:" + qa_id, "docx_uri", docx_download_url)

    return gr.DownloadButton(
        value=sanitize_url(docx_download_url), visible=True
    ), gr.DownloadButton(value=sanitize_url(pdf_download_url), visible=True)


def climate_chat(
    message,
    history,
    questions,
    answers,
    rag_filter,
    improve_question,
    do_rerank,
    language,
    initial_generation,
):
    happy_with_answer = True
    getting_feedback = False
    answer = ""
    if len(history) > 1:
        getting_feedback = history[-2][1] == "Are you happy with the answer? (y/n)"
        if getting_feedback:
            if message.lower() == "n":
                happy_with_answer = False
            else:
                happy_with_answer = True
                getting_feedback = False
        if happy_with_answer:
            yield "Great! I'm glad I could help. What else would you like to know?", questions, ""

    if getting_feedback:
        message = questions[-1]
    else:
        questions.append(message)
    if rag_filter == "":
        rag_filter = None
    else:
        rag_filter = f"*{rag_filter}*"
    if (getting_feedback and not happy_with_answer) or not getting_feedback:
        for key, value in run_query(
            message,
            db,
            llm="gpt-4o",
            mode="gui",
            rag_filter=rag_filter,
            improve_question=improve_question,
            language=language,
            do_rerank=do_rerank,
            history=history,
            initial_generation=happy_with_answer,
        ):
            if key == "improve_question":
                if improve_question:
                    yield f"""**Improved question:** {value["question"]}""" + (
                        """

                    **Better question (en):** {value["question_en"]}"""
                        if language != "en"
                        else ""
                    ), questions, answers
                else:
                    yield None, questions, answers
            elif key == "formulate_query":
                yield f"""**Generated search queries:**\n\n""" + "\n".join(
                    [
                        (
                            f" * {query.query} ({query.query_en})"
                            if language != "en"
                            else f" * {query.query_en}"
                        )
                        for query in value["search_prompts"]
                    ]
                ), questions, answers

            elif key == "retrieve_from_database":
                yield "**Retrieving documents from database...**", questions, answers
                # yield f"""Search queries: {value["search_prompts"]}

                # Retrieved from database: {[doc.metadata["id"] for doc in value["documents"]]}""", questions
            elif key == "web_search_node":
                yield f"""**Searching the web for more information...**""", questions, answers
                # yield f"""Search query: {value["search_query"]}

                # Search query (en): {value["search_query_en"]}""", questions
            elif key == "rerank_documents":
                # yield f"""Reranked documents: {[doc.metadata["id"] for doc in value["documents"]]}""", questions
                yield f"""**Reranking documents...**""", questions, answers

            elif key == "generate":
                answers.append(value["qa_id"])
                answer = compile_answer(
                    value["generation"],
                    value["initial_question"],
                    [doc.metadata.get("source") for doc in value["documents"]],
                )

                yield answer, questions, answers
            elif key == "add_urls_to_database":
                yield f"""**Added new pages to database**""", questions, answers
            elif key == "ask_user_for_feedback":
                yield f"""Are you happy with the answer? (y/n)""", questions, answers
            else:
                yield str(value), questions, answers


with gr.Blocks(
    title="Climate RAG",
    fill_height=True,
    css="""
.h-full {
    height: 85vh;
}
.h-20 {
    height: 20px;
}
.scroll-y {
overflow-y: scroll;
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
.xterm-viewport {
width: max-content;
height: max-content;
}

""",
) as demo:
    # Define how to store state
    chat_state = gr.State([])
    questions_state = gr.State([])
    answers_state = gr.State([])

    gr.Markdown("# Climate RAG")
    with gr.Tab("Chat"):
        # Add a header
        with gr.Row(elem_classes=["h-full"]):
            with gr.Column(variant="panel"):
                improve_question_checkbox = gr.Checkbox(
                    value=True, label="Auto-improve question?"
                )
                do_initial_generation_checkbox = gr.Checkbox(
                    value=True, label="Generate answer before web search?"
                )
                do_rerank_checkbox = gr.Checkbox(value=True, label="Rerank documents?")
                language_dropdown = gr.Dropdown(
                    choices=[
                        ("English", "en"),
                        ("Chinese", "zh"),
                        ("Vietnamese", "vi"),
                        ("Japanese", "ja"),
                        ("Indonesian", "id"),
                        ("Korean", "ko"),
                        ("Russian", "ru"),
                        ("Kazakh", "kk"),
                        ("Italian", "it"),
                        ("Spanish", "es"),
                        ("German", "de"),
                    ],
                    label="Language",
                    value="en",
                    filterable=True,
                )
                rag_filter_textbox = gr.Textbox(
                    placeholder="eg. carbontracker.org or shell*annual",
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

            with gr.Column(scale=4):
                chat_header = gr.Markdown("## Chat")
                chatbot = gr.Chatbot(elem_id="chatbot", scale=4, show_label=False)
                with gr.Row():
                    chat_input = gr.MultimodalTextbox(
                        elem_id="msg-box",
                        interactive=True,
                        placeholder="Ask a question. e.g. What does the Shell 2023 annual report say about climate change?",
                        file_types=[],
                        show_label=False,
                        scale=6,
                    )
                with gr.Row():
                    stop_button = gr.Button(value="Stop", variant="stop", scale=5)
                    download_word_button = gr.DownloadButton(
                        icon="static/ri--file-word-line.svg",
                        label="Download DOCX",
                        size="sm",
                        scale=1,
                        visible=False,
                    )
                    download_pdf_button = gr.DownloadButton(
                        icon="static/ri--file-pdf-line.svg",
                        label="Download PDF",
                        size="sm",
                        scale=1,
                        visible=False,
                    )
                    clear = gr.ClearButton([chat_input, chatbot], scale=5)
    with gr.Tab(
        "Previous queries", elem_classes=["h-full", "scroll-y"]
    ) as previous_queries_tab:
        with gr.Row():
            query_history_display = gr.Dataset(
                components=[
                    gr.Textbox(visible=False),
                    gr.Markdown(visible=False),
                    gr.Markdown(visible=False, elem_classes=["scroll-y", "h-20"]),
                    gr.HTML(visible=False),
                    gr.HTML(visible=False),
                ],
                label="History",
                headers=["Date", "Question", "Answer", "PDF", "DOCX"],
            )

    with gr.Tab("Documents", elem_classes=["h-full"]):
        with gr.Row():
            gr.Markdown("## Add new documents")
        with gr.Row():
            with gr.Column(scale=1):
                new_file = gr.File(
                    label="Upload documents",
                    file_types=["pdf", "PDF", "md", "MD"],
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
            search_input = gr.Textbox(placeholder="Search documents", show_label=False)
            search_button = gr.Button(value="Search", visible=False)
        with gr.Row():
            search_results_display = gr.Dataset(
                components=[
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Number(visible=False),
                ],
                label="Documents retrieved",
                headers=["Source", "Date added", "Page length"],
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
        rag_filter,
        improve_question,
        do_rerank,
        language,
        initial_generation,
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
        bot_messages = climate_chat(
            message=message,
            history=chat_history,
            questions=questions,
            answers=answers,
            rag_filter=rag_filter,
            improve_question=improve_question,
            do_rerank=do_rerank,
            language=language,
            initial_generation=initial_generation,
        )
        for bot_message, questions, answers in bot_messages:
            chat_history.append([None, bot_message])
            yield chat_history, chat_history, questions, answers, *download_latest_answer(
                questions, answers
            ), f"## Chat\n### {questions[-1]}"

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
            rag_filter_textbox,
            improve_question_checkbox,
            do_rerank_checkbox,
            language_dropdown,
            do_initial_generation_checkbox,
        ],
        [
            chatbot,
            chat_state,
            questions_state,
            answers_state,
            download_word_button,
            download_pdf_button,
            chat_header,
        ],
    )

    def stop_querying(questions):
        # Remove the last question
        questions.pop()
        return questions, "## Chat"

    stop_button.click(
        fn=stop_querying,
        inputs=[questions_state],
        outputs=[questions_state, chat_header],
        cancels=[converse_event],
        queue=False,
    )
    clear.click(
        fn=lambda: (None, []), inputs=None, outputs=[chatbot, chat_state], queue=False
    )

    def filter_documents(search_query):
        from query_data import query_source_documents

        if search_query is None:
            search_query = ""

        if len(search_query) < 2:
            search_results = query_source_documents(db, "", print_output=False)[
                ["source"]
            ]
        else:
            search_results = query_source_documents(
                db, f"*{search_query}*", print_output=False
            )[["source"]]
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
        fn=lambda x: filter_documents(x),
        inputs=[rag_filter_textbox],
        outputs=[doc_sources],
        queue=False,
    )

    ## Tab 2: Query history
    def get_query_history(evt: gr.EventData):
        import pandas as pd
        import msgspec
        import datetime

        all_answers = []
        for key in r.keys("*:answer:*"):
            answer = r.hgetall(key)
            answer["sources"] = msgspec.json.decode(answer["sources"])
            answer["doc_ids"] = msgspec.json.decode(answer["doc_ids"])
            answer["date_added_ts"] = humanize.naturaltime(
                datetime.datetime.now(datetime.UTC)
                - datetime.datetime.fromtimestamp(
                    int(answer["date_added"]), tz=datetime.UTC
                )
            )
            if "pdf_uri" in answer.keys():
                answer["pdf_uri"] = sanitize_url(answer["pdf_uri"])
            if "docx_uri" in answer.keys():
                answer["docx_uri"] = sanitize_url(answer["docx_uri"])
            all_answers.append(answer)

        df = (
            pd.DataFrame.from_records(all_answers)
            .sort_values("date_added", ascending=False)
            .reindex(
                ["date_added_ts", "question", "answer", "pdf_uri", "docx_uri"], axis=1
            )
            # .iloc[:20]
        )
        df.pdf_uri = (
            "<a target='_blank' href='"
            + df.pdf_uri.astype(str).fillna("")
            + "'><img src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTUgNGgxMHY0aDR2MTJINXpNMy45OTkgMkEuOTk1Ljk5NSAwIDAgMCAzIDIuOTkydjE4LjAxNmExIDEgMCAwIDAgLjk5My45OTJoMTYuMDE0QTEgMSAwIDAgMCAyMSAyMC45OTJWN2wtNS01em02LjUgNS41YzAgMS41NzctLjQ1NSAzLjQzNy0xLjIyNCA1LjE1M2MtLjc3MiAxLjcyMy0xLjgxNCAzLjE5Ny0yLjkgNC4wNjZsMS4xOCAxLjYxM2MyLjkyNy0xLjk1MiA2LjE2OC0zLjI5IDkuMzA0LTIuODQybC40NTctMS45MzlDMTQuNjQ0IDEyLjY2MSAxMi41IDkuOTkgMTIuNSA3LjV6bS42IDUuOTcyYy4yNjgtLjU5Ny41MDUtMS4yMTYuNzA1LTEuODQzYTkuNyA5LjcgMCAwIDAgMS43MDYgMS45NjZjLS45ODIuMTc2LTEuOTQ0LjQ2NS0yLjg3NS44MzNxLjI0OC0uNDcxLjQ2NS0uOTU2Ii8+PC9zdmc+' alt='PDF'></img></a>"
        )
        df.docx_uri = (
            "<a target='_blank' href='"
            + df.docx_uri.astype(str).fillna("")
            + "'><img src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTE2IDh2OGgtMmwtMi0ybC0yIDJIOFY4aDJ2NWwyLTJsMiAyVjhoMVY0SDV2MTZoMTRWOHpNMyAyLjk5MkMzIDIuNDQ0IDMuNDQ3IDIgMy45OTkgMkgxNmw1IDV2MTMuOTkzQTEgMSAwIDAgMSAyMC4wMDcgMjJIMy45OTNBMSAxIDAgMCAxIDMgMjEuMDA4eiIvPjwvc3ZnPg==' alt='DOCX'></img></a>"
        )

        return gr.Dataset(samples=df.to_numpy().tolist())

    previous_queries_tab.select(
        get_query_history, outputs=[query_history_display], queue=False
    )

    ## Tab 3: Documents
    # Search documents
    def search_documents(search_query):
        from query_data import query_source_documents

        if search_query is None:
            search_query = ""

        if len(search_query) < 3:
            search_results = query_source_documents(db, "", print_output=False)[
                ["source", "date_added", "page_length", "page_content"]
            ]
        else:
            search_results = query_source_documents(
                db, f"*{search_query}*", print_output=False
            )[["source", "date_added", "page_length", "page_content"]]
        return gr.Dataset(samples=search_results.to_numpy().tolist())

    search_input.change(
        fn=search_documents,
        inputs=[search_input],
        outputs=[search_results_display],
        queue=False,
    )
    search_button.click(
        fn=search_documents,
        inputs=[search_input],
        outputs=[search_results_display],
        queue=False,
    )

    # Add new documents
    def add_document(url):
        from tools import add_urls_to_db

        add_urls_to_db([url], db)

        # Retrieve source markdown
        page_content = query_source_documents(db, f"*{url}*", print_output=False)[
            "page_content"
        ]
        if len(page_content) > 0:
            page_content = page_content.iloc[0]
        else:
            page_content = f"Error loading document: {url}. See console for details."
        return page_content

    add_button.click(
        fn=add_document,
        inputs=[url_input],
        outputs=[selected_source],
        queue=False,
    )
    # .then(fn=lambda: None, inputs=None, outputs=[url_input], queue=False).then(
    #     search_documents,
    #     [search_input],
    #     [search_results_display],
    # )

    # Upload new document
    from tools import upload_documents

    new_file.upload(
        fn=lambda x: upload_documents(x, db)[-1],  # Return the last document added only
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


demo.queue()
demo.launch(inbrowser=False, show_api=False)
