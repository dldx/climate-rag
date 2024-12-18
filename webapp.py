import logging
import re
import urllib.parse
import pandas as pd
from pathvalidate import sanitize_filename
import humanize
import msgspec
from typing import List, Tuple
import gradio as gr
from dotenv import load_dotenv
import os
import shutil
import hashlib
from langchain_core.documents import Document
from gradio_log import Log
from ulid import ULID
import sys
from constants import language_choices

sys.settrace(None)

load_dotenv()

from helpers import (
    clean_urls,
    compile_answer,
    generate_qa_id,
    render_qa_pdfs,
    sanitize_url,
)
from query_data import query_source_documents, run_query
from tools import get_vector_store
from cache import r

db = get_vector_store()


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
                yield "Great! I'm glad I could help. What else would you like to know?", questions, answers
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
        llm="gemini-2.0-flash-exp",
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
    ):
        if key == "improve_question":
            if improve_question:
                yield f"""**Improved question:** {value["question"]}""" + (
                    f"""

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
        elif key == "add_additional_metadata":
            yield f"""**Fetching document titles, entities, etc...**""", questions, answers
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
            # After generation ask for feedback
            yield USER_FEEDBACK_QUESTION, questions, answers
        elif key == "add_urls_to_database":
            yield f"""**Added new pages to database**""", questions, answers
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

    gr.HTML(
        "<div align='center'><img alt='Climate RAG logo' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODU5LjM5IiBoZWlnaHQ9IjMyNi43NyIgdmVyc2lvbj0iMS4xIiB2aWV3Qm94PSIwIDAgODU4LjM5IDMyNi4zOSIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJtMjgzLjU4IDE3LjA1NmMtNy44NDUyLTAuMTg1NjItMTYuMDk5IDAuODc3MDEtMjIuMTc0IDAuODc3MDFoLTExMy4zM2MwIDE3LjYzMiA0Ljk0NTcgNDMuNTc4LTE0LjI1NSA1My4xODYtMTEuNDI2IDUuNzE2OS0yOS41NzMgMi40MDgzLTQyLjA1MiAyLjQwODN2NTguNDQ1YzAgNi4zMTI4LTIuMzQ0NSAxOC4zMzIgMC40Mjg3NSAyNC4wMDYgMS4xMDU1IDIuMjYwOCA1LjM1MyAyLjI1NTQgNy40MTE0IDMuMzc0NCAzLjY3MzUgMS45OTg1IDYuNjQzNCA1LjQ2MzkgOC44MjQ0IDguOTY5MSA5LjU0NjUgMTUuMzQgMTguMDQ1IDMxLjQxMyAyNy4xMTUgNDcuMDQxIDYuNDgzOCAxMS4xNzIgMTQuMzM5IDIyLjMxOCAxOS40MTQgMzQuMjEyIDMuNjk3IDguNjY0OC0wLjczNDIxIDE4LjM5LTkuMDI0OCAyMi41MzgtNi4wNzExIDMuMDM3LTEzLjM4MSAyLjQwODMtMTkuOTU3IDIuNDA4M2gtMzQuMjEyYzAgMTAuMTEyLTIuMjMwMSAyMy44OTQgNy4xNTUzIDMwLjc5IDkuODQ2NSA3LjIzNDMgMjkuNjY5IDMuNDIxNyA0MS4zMTEgMy40MjE3aDE0NS40YzguMzA3IDAgMTYuNTA2LTEuNDc5IDIwLjI5Mi05Ljk3ODQgNC44MjUzLTEwLjgzMyAxLjgwMjgtMjcuNTI5IDEuODAyOC0zOS4yMDF2LTMwLjY0OGMwLTQuMjY1OCAxLjA1ODgtMTAuMjE2LTAuNDMwMTctMTQuMjU1LTEuNTMwMi00LjE1MDMtNi40OTQyLTcuNjM3LTkuNTQ4Mi0xMC42OTFsLTE5LjI0NC0xOS4yNDRjLTIuNzcxMS0yLjc3MTEtNS43NDEyLTYuOTAxNC05LjI2NTYtOC43MDQ2LTMuMTQ2LTEuNjEwMS03LjE3NTkgMC43MDgwOC0xMC42OTEtMC42NjY3OS0zLjk3MjEtMS41NTM4LTYuMDQzNC02LjE3MTYtOS45Nzg0LTcuNDgzOC0zLjA0ODQtMS4wMTU3LTcuNjY5OCAxLjYyMDUtMTAuNjkxIDIuMjk2OS02Ljc5NzQgMS41MjE3LTEzLjcwNiAyLjE0MzItMjAuNjcgMS42NjM1LTMxLjIwMS0yLjE0ODItNTguMjEzLTI3LjcxNi02My4wMTQtNTguMzc5LTEuNDE2OS05LjA1MTEtMS4yMjk5LTE4LjE4MyAxLjExNS0yNy4wODQgMTQuOTMzLTU2LjY4NSA4OS4wNDMtNzAuNzg0IDEyMy41Ni0yMy41MjEgMTAuMTM3IDEzLjg4MiAxNC4yMzUgMzEuNTg1IDEyLjM0NiA0OC40NjctMC42MzcxOCA1LjY5NzYtNS4wNzg4IDEzLjA1Ni00LjQ5MzYgMTguNTMxIDAuNDE2MjYgMy44OTU5IDUuMjA4NyA2LjQ1NDYgNi43NzI0IDkuOTc4NCAxLjU3NTIgMy41NDk1LTAuMzU1NTggNi41OTE1IDAuNTMxNzggOS45Nzg0IDEuMzIxNCA1LjA0NDEgOS41NzY4IDExLjM0OSAxMy43MDIgMTQuMjU1di0xMDEuMjFjMC0xMS4zMDggMy42NjIxLTMwLjA5MS00LjQyODItMzkuMTczLTQuNDQzMi00Ljk4OC0xMS44OC02LjQyMTItMTkuNzI2LTYuNjA2OHptLTE0Mi42MyA1LjE1MzVjLTE0LjUzNSAxNC41MzUtMzMuMDc2IDI4LjgyNi00NC45MDMgNDUuNjE2IDEzLjUzNiAwIDM4Ljc5NCA0Ljk4MiA0NC44MDgtMTEuNDA0IDIuMDcxOS01LjY0NTUgMC44MDc0MS0xMy4zMDIgMC44MDc0MS0xOS4yNDQgMC00LjU3NiAxLjA2Ny0xMC43MjYtMC43MTI3NC0xNC45Njh6bTgwLjcwNyAxNi45NjRjLTEuOTgzMSAwLjAzNzY0LTMuOTQ3NCAwLjE4MjUxLTUuODY5IDAuNDQ0MDctMTAuNjYzIDEuNDUxNC0yMC4yMTYgNC42MzM1LTI5LjIyMiAxMC42NDEtNi42ODcgNC40NjAyLTEyLjQwNiAxMC4xNTQtMTYuODUyIDE2Ljg1NC0yOS4zMjQgNDQuMTc5IDYuOTA3NyAxMDUuNDMgNjAuMzMgOTguODY3IDkuMjY5Mi0xLjEzODIgMTguNDU2LTQuMjU4MiAyNi4zNzEtOS4yMjUzIDQ2LjExMy0yOC45NDIgMzUuMDgyLTEwMC44Mi0xNy4xMDYtMTE1LjI5LTUuNTg5My0xLjU0OTctMTEuNzAyLTIuNDAwMS0xNy42NTItMi4yODcyem0yNi4yODUgMzEuNjU2YzMuNzk5NyAwLjEwMjAxIDcuNDY3NyAwLjczNjY2IDguNDUxMyAyLjIxMDYgMS43Njc2IDIuNjQ4NiAwLjczMzYzIDguMTI0MyAwLjczMzYzIDExLjE3OHYyOS4yMjJjMCAzLjk5MjggMi4wODggMTYuOTIyLTIuMjQ0IDE4LjgxNC0zLjAyMTMgMS4zMTkzLTE1LjcyNiAxLjcyNTctMTcuNjkzLTEuMjIyMi0xLjc2NzYtMi42NDg2LTAuNzMyMjItOC4xMjI5LTAuNzMyMjItMTEuMTc3di0yOS4yMjJjMC0zLjk5MjgtMi4wODk0LTE2LjkyNCAyLjI0MjYtMTguODE1IDEuNTEwNy0wLjY1OTY0IDUuNDQyMy0xLjA5MDQgOS4yNDItMC45ODgzN3ptLTUuNzgyNyA2LjI2MTZ2NDkuODkyaDkuMjY1NnYtNDkuODkyem0tMjAuMjgzIDUuMTc5OWMzLjU1MjcgMC4wODkzMSA2Ljk2MjMgMC42OTM4NyA4LjAzNzggMi4yNTY2IDEuNTYzIDIuMjcxNSAwLjg0MDgyIDYuMzQ3MyAwLjg0MDgyIDguOTU2NnYyMC42N2MwIDQuMTIzOSAyLjE3MiAxNi45MTMtMi44OTQxIDE4LjMwNC0zLjIyOTQgMC44ODY2Ni0xNS4wMTQgMS42MTU4LTE3LjA0My0xLjQyNTUtMS40MTI2LTIuMTE2MS0wLjczMjIxLTUuOTEwNS0wLjczMjIxLTguMzI2di0yMC42N2MwLTQuMjgxNC0yLjI1MDQtMTcuMzk1IDIuODkyNy0xOC45MzUgMS42NDk2LTAuNDk0MjkgNS4zNDU0LTAuOTIwMzQgOC44OTgyLTAuODMxMDd6bS02LjA4ODkgNi4yMjR2MzguNDg4aDkuMjY1NnYtMzguNDg4em0tMTguODAzIDcuNzI0NmMyLjg1NzggMC4xODAyMyA1LjU2NDQgMC44MjYwOCA2LjY2NTMgMi40NzkzIDIuNTA4OCAzLjc2ODMgMC43MzM2MiAxMy4xNjQgMC43MzM2MiAxNy41OTMgMCA0LjI4NzkgMi4xMDkxIDE1LjEyMS0zLjU5NTcgMTYuMjMtMy4zMSAwLjY0MjktMTQuMjcgMS42MTU5LTE2LjM0Mi0xLjQ4OTUtMS4zMTc5LTEuOTc0My0wLjczMjIyLTUuMzYzOS0wLjczMjIyLTcuNjEzMyAwLTYuNDg3NC0yLjQyMDEtMTguODkyIDAuNDI4NzUtMjQuNzIgMC44ODk1LTEuODE4MiAzLjUxMjctMS45NDAzIDUuMjczMi0yLjE2MDUgMS43MDI3LTAuMjEzMTIgNC43MTA5LTAuNDk5MDMgNy41Njg3LTAuMzE4OHptLTcuNTY4NyA2LjUzMDJ2MjQuMjMzaDkuMjY1NnYtMjQuMjMzem05My4zNjkgMzMuNDk5Yy02LjQ2MSAxMi4wOS0xNi44ODIgMjAuOTM4LTI4LjUxIDI3Ljc5NyA3LjA4ODkgMTUuMTU3IDE5Ljk4OS0yLjE2OTYgMjYuMzcxLTguNTUyOSAyLjc4MTEtMi43ODE4IDcuMDIzLTUuOTA1MSA3Ljg1NTUtOS45Nzg0IDAuODYwOTgtNC4yMTIzLTIuMjEwNi03LjYxNTctNS43MTczLTkuMjY1NnptNy4xMjc0IDE5LjM5Yy01LjA4NzYgMC0xMy42MjEgMTIuMDQ5LTE3LjEwNiAxNS41MzQgNS43NTk3IDguMTc2NiAxNC4zMTMgMTUuMDI2IDIxLjM4MiAyMi4wOTVsMTQuOTY4IDE0Ljk2OGM0LjE5MTYgNC4xOTE2IDkuMTEzOSAxMS4yNzEgMTQuOTY4IDEzLjA3NiA2LjgzNDUgMi4xMDc2IDE2Ljc3My02Ljg0IDE2LjA5MS0xMy43ODktMC41MjEwMi01LjMwNzgtNi4xOTEtOS4zNDQxLTkuNjc2My0xMi44MjlsLTI3Ljc5Ny0yNy43OTdjLTIuNzQ0MS0yLjc0NC04LjY4NTUtMTEuMjU4LTEyLjgyOS0xMS4yNTh6bS0xOTguMzUgNy40NTg3Yy0xLjI1OTcgMC4wMjEyLTIuNjEwOCAwLjI0MzI2LTQuMDY3NiAwLjcwMDIyLTcuNDE0NiAyLjMyNTctMTEuMzA1IDEzLjI5NS0xNC44NjIgMTkuNDkyLTEwLjUxOCAxOC4zMjQtMjAuOTc4IDM2LjcwNS0zMS43NDkgNTQuODgxLTQuMjM0OCA3LjE0Ni0xMS43NjYgMTYuNTc0LTUuOTQ2OSAyNC44NzkgNS4zMDg3IDcuNTc2NCAxNy40NTggNS4wNTYgMjUuNDc0IDUuMDU2aDQ2Ljg4NWMtMS44NzEtMi4yMTYtMy41NjktNC4zNDY3LTQuODAxMy01LjkyNjEtMTAuMTg3IDEuNjg2My0xOC40ODMtNS43NzM4LTIyLjcyOC0xNi4xMDItNy4yNTYyLTE3LjU4OCAzLjczMjUtNjIuNzI3IDYuMjI2OC01NS43ODIgOC4zNDI0IDIzLjIxOCAyMy45MTggMzEuMDUzIDMwLjI3OSA0MS42MzMgNS45MDUxIDkuODEwOC0wLjkwMzUxIDIzLjgxMS02Ljk3MjkgMjcuMzU4LTEuNjg2LTIuMTg1Ni0zLjg0ODctNS4wMzQ1LTYuMTk2MS04LjQ5NDQtMi41NTgzLTUuMTQ4OCAxLjA2NzctMTkuMjY2IDEuMDY3Ny0xOS4yNjYtMy4wNjc5IDYuMDgwNy00LjA1NDQgMTAuMDE5LTQuMTkwMSAxMi45ODQtMi4wMTU1LTMuMzA4LTQuMzM2Mi03LjQ4NTMtNi4zMDg5LTExLjY5MSAwLjMyMjk0IDVlLTMgLTAuMDM3Mi0wLjQ0NjEyIDIuNzg2OS0xNi40NSAwIDAtNC43MjE3IDYuNjc4My00LjU3MTYgMTEuNi0yLjY4OTYtNi45MzczLTQuOTEwMi0xNS4xMDktNi4wNDg2LTI0Ljg0MiAwIDAtMi4wOTIgMTQuMDg5IDUuNjk5MiAzMy4zMTQtMS43NDE1LTAuNTcyMzMtNi41ODY2LTIuOTMzNS0xMC4wNy03LjQ4OCAxLjI0MDIgNS4zMjEgNi41NTY1IDkuNzQyOCAxMi40MSAxMi41OTggMi45NjggNi4zODE0IDYuNDk2IDEyLjk3OCAxMi4wOTQgMTkuNzE2LTAuMDE4MiA2ZS0zIC0wLjAzNzQgMC4wMTE2LTAuMDU1NiAwLjAxODEgMy43OTcyIDQuMzI1NSA4LjA1MDggNi4xNjUgOS45Nzg0IDYuODE5OGgxLjUzNTVjNy40NTAzIDAgMTguNzA4IDIuMDk4NSAyNS42NTktMC43MTk2OSAxMS4xNjQtNC41Mjc0IDcuMzMzNi0xNS44NjcgMi44NDI2LTIzLjUxNC0xMi41MDYtMjEuMjkzLTI0LjU1Ny00Mi44NTQtMzcuMDYzLTY0LjE0Ny00LjE4NjYtNy4xMjg5LTguNDg4NS0xNi43NzctMTcuMzA2LTE2LjYyOHptMTA0LjU2IDQxLjc1NWM0LjUzNjYgMC4xNTU5MiA5LjA5NiAwLjUzMDM0IDEzLjI0NyAwLjUzMTc3IDIuNTMwOSA3ZS00IDYuODIzNi0wLjYwMDM1IDguNTMyIDEuNzkxNiAxLjQzNjIgMi4wMDk5IDAuNzMzNjMgNS44NTQ3IDAuNzMzNjMgOC4xODY4djIxLjM4MmMwIDIuNTQ4OCAwLjk0MTIgNy4zODAyLTIuMjQ0IDguMzI2LTcuODkwOCAyLjM0MjgtMTkuNDQ4IDAuMjI2OTEtMjcuNjkxIDAuMjI2OTEtMi4yNDU5IDAtNi45NDE5IDAuODY2MzItOC41MzM0LTEuMTQyOS0xLjU3MDItMS45ODI5LTAuNzMyMjItNi40NjM1LTAuNzMyMjItOC44MzU1di0yMi4wOTVjMmUtMyAtMy4zMTI4LTAuMzkyMzEtNi43NDQxIDMuNTc2Mi03LjU5NTIgNC4wNjA1LTAuODcwMjYgOC41NzU0LTAuOTMyNyAxMy4xMTItMC43NzY3OHptLTEwLjk4NiA2Ljk0NjR2MjcuNzk3aDI3Ljc5N3YtMjcuNzk3em01NS41MzQgMi40NDg3YzIuNTY4OSAwLjA5OCA1LjE3NjMgMC40MDIzMSA3LjE4NzMgMC40MDIzMWgzMi4wNzNjMS44OTIzIDAgMTYuMTQxLTAuNjk5MzMgMTAuOTcgNC40NjAyLTIuNjY1NiAyLjY1OTItMTAuMzUyIDEuMjQxNy0xMy44MjEgMS4yNDE3LTEzLjI2MSAwLTI5LjAyMSAyLjM2NTItNDIuMDIxLTAuMTYyODYtMi42MjcyLTAuNTExMDQtNC41NTg0LTMuNjkxMy0xLjM0ODktNS4xMTAzIDEuODU5OS0wLjgyMTc4IDQuMzkxNS0wLjkyOTA2IDYuOTYwNC0wLjgzMTA2em0tMS43MjQ4IDE2LjUyN2MxLjg5NTcgMC4wNTM0IDMuODIwMiAwLjI2ODY3IDUuMzQ4NCAwLjI2ODY3IDcuNTU1MSAwIDE1LjI2NS0wLjQ0OTc2IDIyLjgwNCAwLjAzMDcgMi4yOSAwLjE0NjEyIDYuMjQxIDIuNDc5NCAzLjEzNjQgNC45MzkxLTEuOTgxNCAxLjU2OTQtNi40NjI4IDAuNzMyMjMtOC44MzQxIDAuNzMyMjMtOC4xNzQ0IDAtMTkuMTcgMS45NDctMjcuMDQyLTAuMjI2OTEtMi41OTE1LTAuNzE1NTktMy4yNTg3LTMuODM0Ni0wLjU5MTYyLTUuMDQ2MyAxLjQxNjktMC42NDMyNCAzLjI4NDItMC43NTA3OSA1LjE3OTktMC42OTc0NHptLTQ1LjAxNCAyMi4yOTNjNS4zNDk4IDAuMTQ0NDUgMTAuODgzIDAuNzgzMzQgMTUuNDM4IDAuNzgzNzQgMi41NDA5IDAgNi44NDg1LTAuNzEyMzIgOC41MzIgMS43MzQ1IDEuNDIxOSAyLjA2NTUgMC43MzM2MyA1Ljg2OSAwLjczMzYzIDguMjQzOXYyMS4zODJjMCAyLjU0ODggMC45NDEyIDcuMzgwMi0yLjI0NCA4LjMyNi03Ljg5MDggMi4zNDI4LTE5LjQ0OCAwLjIyNjktMjcuNjkxIDAuMjI2OS0yLjI0NTkgMC02Ljk0MTkgMC44NjYzMy04LjUzMzQtMS4xNDI5LTEuNDc4Mi0xLjg2Ni0wLjczMjIyLTUuODk3Ni0wLjczMjIyLTguMTIyOHYtMjIuMDk1YzAtMi45ODA3LTAuNzEzNzQtNy4xNjM0IDIuODkyNy04LjI0MzkgMS45Mzk2LTAuNTgxMjUgNC4wOTI4LTAuODkyMTEgNi4zNTktMS4wMzAyIDEuNjk5Ni0wLjEwMzUxIDMuNDYyMS0wLjExMDc4IDUuMjQ1My0wLjA2MjZ6bS04Ljc5NTEgNy4xOTg0djI3Ljc5N2gyNy43OTd2LTI3Ljc5N3ptNTUuNTM0IDIuNDQ4N2MyLjU2ODggMC4wOTggNS4xNzYzIDAuNDAyMzEgNy4xODczIDAuNDAyMzFoMzIuMDczYzEuODkyMyAwIDE2LjE0MS0wLjY5OTMyIDEwLjk3IDQuNDYwMi0yLjY2NTYgMi42NTkyLTEwLjM1MiAxLjI0MTctMTMuODIxIDEuMjQxNy0xMy4yNjEgMC0yOS4wMjEgMi4zNjUyLTQyLjAyMS0wLjE2Mjg3LTIuNjI3Mi0wLjUxMTAzLTQuNTU4NC0zLjY5MTMtMS4zNDg5LTUuMTEwMyAxLjg2MDItMC44MjE3OCA0LjM5MTYtMC45MjkwNyA2Ljk2MDQtMC44MzEwNnptLTEuNzQ5OCAxNi41MzRjMS44OTggMC4wNTEyIDMuODIzNiAwLjI2MTcxIDUuMzczNCAwLjI2MTcxIDcuNzM3NSAwIDE1LjgwMi0wLjY1OTU4IDIzLjUwNyAwLjA0MTcgMi4zMzQyIDAuMjEyNCA1LjA4MDUgMi44MTU0IDIuNDMzNCA0LjkyNjYtMS45NzIyIDEuNTczLTYuNDcgMC43MzM2MS04LjgzNDEgMC43MzM2MS04LjE4MDEgMC0xOS4xNjIgMS45NDEzLTI3LjA0Mi0wLjIyNjktMi42MDY1LTAuNzE3MDItMy4zOTk5LTMuODM2LTAuNjQ4NzEtNS4wNDYzIDEuNDQzMy0wLjYzNDM0IDMuMzE0LTAuNzQxNjEgNS4yMTE5LTAuNjkwNDh6IiBmaWxsPSIjODliMjY0Ii8+PGcgZmlsbD0iIzg5YjI2NCIgc3Ryb2tlLXdpZHRoPSIuOTk5OTkiIGFyaWEtbGFiZWw9IkNsaW1hdGUgUkFHIj48cGF0aCBkPSJtNDU2LjI3IDE0Mi4yM3EtMi4yMDgyIDEuNjU2Mi02LjYyNDcgMy44NjQ0LTQuMjc4NSAyLjA3MDItMTAuMjEzIDMuNTg4NC01LjkzNDYgMS41MTgyLTEyLjk3MyAxLjM4MDItMTEuNzMxLTAuMTM4MDItMjEuMTE2LTQuMTQwNC05LjI0Ny00LjAwMjQtMTUuODcyLTEwLjkwMy02LjQ4NjctNy4wMzg4LTkuOTM3MS0xNi4wMS0zLjQ1MDQtOS4xMDktMy40NTA0LTE5LjMyMiAwLTExLjMxNyAzLjU4ODQtMjAuODR0MTAuMjEzLTE2LjQyNHE2LjYyNDctNy4wMzg4IDE1LjU5Ni0xMC45MDMgOS4xMDktMy44NjQ0IDIwLjAxMi0zLjg2NDQgOS4yNDcgMCAxNi43IDIuNDg0M3QxMi41NTkgNS42NTg2bC01LjkzNDYgMTQuMjE2cS00LjAwMjQtMi43NjAzLTkuOTM3MS01LjI0NDYtNS43OTY2LTIuNjIyMy0xMi45NzMtMi42MjIzLTYuOTAwOCAwLTEzLjExMSAyLjc2MDMtNi4yMTA3IDIuNzYwMy0xMS4wNDEgNy44NjY5LTQuNjkyNSA0Ljk2ODUtNy40NTI4IDExLjU5My0yLjYyMjMgNi42MjQ3LTIuNjIyMyAxNC40OTJ0Mi40ODQzIDE0LjYzcTIuNDg0MyA2Ljc2MjcgNy4wMzg4IDExLjczMSA0LjU1NDUgNC44MzA1IDExLjA0MSA3LjU5MDggNi40ODY3IDIuNzYwMyAxNC40OTIgMi43NjAzIDcuNzI4OCAwIDEzLjUyNS0yLjM0NjIgNS45MzQ2LTIuMzQ2MyA5LjY2MS01LjM4MjZ6IiBzdHlsZT0iZm9udC12YXJpYXRpb24tc2V0dGluZ3M6J3dnaHQnIDUyMiIvPjxwYXRoIGQ9Im00NzQuOSA0Mi41ODdoMTQuNDkydjEwNy4zOGgtMTQuNDkyeiIgc3R5bGU9ImZvbnQtdmFyaWF0aW9uLXNldHRpbmdzOid3Z2h0JyA1MjIiLz48cGF0aCBkPSJtNTExLjA2IDkxLjk5N2gxNC40OTJ2NTcuOTY2aC0xNC40OTJ6bS0wLjY5MDA4LTIwLjcwMnEwLTMuNDUwNCAyLjYyMjMtNS42NTg2IDIuNzYwMy0yLjIwODIgNS43OTY2LTIuMjA4MiAzLjE3NDMgMCA1LjY1ODYgMi4zNDYzIDIuNDg0MyAyLjIwODIgMi40ODQzIDUuNTIwNiAwIDMuNDUwNC0yLjQ4NDMgNS42NTg2LTIuNDg0MyAyLjA3MDItNS42NTg2IDIuMDcwMi0zLjAzNjMgMC01Ljc5NjYtMi4wNzAyLTIuNjIyMy0yLjIwODItMi42MjIzLTUuNjU4NnoiIHN0eWxlPSJmb250LXZhcmlhdGlvbi1zZXR0aW5nczond2dodCcgNTIyIi8+PHBhdGggZD0ibTU2MC4zMyA5MS45OTcgMS4yNDIxIDEyLjI4My0wLjU1MjA2LTAuOTY2MXEzLjE3NDQtNi40ODY3IDguOTcxLTEwLjA3NXQxMy4zODctMy41ODg0cTQuODMwNSAwIDguNTU2OSAxLjUxODIgMy43MjY0IDEuMzgwMiA2LjIxMDcgNC4yNzg1IDIuNDg0MyAyLjc2MDMgMy4zMTI0IDYuOTAwN2wtMC44MjgwOSAwLjQxNDA1cTMuNTg4NC02LjIxMDcgOS4yNDctOS42NjEgNS43OTY2LTMuNDUwNCAxMi4xNDUtMy40NTA0IDguNTU2OSAwIDEzLjUyNSA0LjgzMDUgNS4xMDY2IDQuODMwNSA1LjI0NDYgMTIuNDIxdjQzLjA2MWgtMTQuMzU0di0zNy41NHEtMC4xMzgwMS00LjI3ODUtMS45MzIyLTcuMTc2OC0xLjc5NDItMi44OTgzLTYuNDg2Ny0zLjE3NDQtNC45Njg1IDAtOC44MzMgMy4xNzQ0LTMuNzI2NCAzLjAzNjMtNS43OTY2IDcuODY2OC0xLjkzMjIgNC44MzA1LTIuMDcwMiAxMC4zNTF2MjYuNDk5aC0xNC40OTJ2LTM3LjU0cS0wLjEzODAyLTQuMjc4NS0yLjA3MDItNy4xNzY4LTEuOTMyMi0yLjg5ODMtNi42MjQ3LTMuMTc0NC00Ljk2ODUgMC04LjY5NDkgMy4xNzQ0LTMuNTg4NCAzLjAzNjMtNS42NTg2IDguMDA0OS0yLjA3MDIgNC44MzA1LTIuMDcwMiAxMC4yMTN2MjYuNDk5aC0xNC40OTJ2LTU3Ljk2NnoiIHN0eWxlPSJmb250LXZhcmlhdGlvbi1zZXR0aW5nczond2dodCcgNTIyIi8+PHBhdGggZD0ibTY4Ni42MiAxNTEuMzRxLTguMTQyOSAwLTE0Ljc2OC0zLjMxMjQtNi42MjQ3LTMuNDUwNC0xMC40ODktMTAuMjEzLTMuODY0NC02Ljc2MjctMy44NjQ0LTE2LjgzOCAwLTkuNjYxIDQuMDAyNC0xNi43IDQuMDAyNC03LjE3NjggMTAuNzY1LTEwLjkwMyA2Ljc2MjctMy44NjQ0IDE0LjYzLTMuODY0NCA4LjI4MDkgMCAxMy41MjUgMy40NTA0IDUuMzgyNiAzLjQ1MDQgOC4yODA5IDguMTQyOWwtMC44MjgwOSAyLjIwODIgMS4zODAyLTExLjMxN2gxMy4zODd2NTcuOTY2aC0xNC40OTJ2LTE0LjQ5MmwxLjUxODIgMy41ODg0cS0wLjU1MjA2IDAuOTY2MTEtMi4zNDYzIDMuMDM2My0xLjc5NDIgMS45MzIyLTQuNjkyNSA0LjE0MDQtMi44OTgzIDIuMjA4Mi02LjkwMDggMy43MjY0LTQuMDAyNCAxLjM4MDItOS4xMDkgMS4zODAyem00LjAwMjQtMTEuODY5cTQuNDE2NSAwIDguMDA0OS0xLjUxODIgMy41ODg0LTEuNjU2MiA1LjkzNDYtNC41NTQ1IDIuNDg0My0zLjAzNjMgMy41ODg0LTcuMTc2OHYtMTIuMTQ1cS0xLjEwNDEtMy43MjY0LTMuNzI2NC02LjQ4NjctMi42MjIzLTIuODk4My02LjM0ODctNC41NTQ1LTMuNTg4NC0xLjY1NjItOC4wMDQ5LTEuNjU2Mi00LjgzMDUgMC04Ljk3MSAyLjQ4NDMtNC4xNDA0IDIuMzQ2Mi02LjYyNDcgNi42MjQ3LTIuMzQ2MiA0LjI3ODUtMi4zNDYyIDkuNzk5MSAwIDUuMzgyNiAyLjQ4NDMgOS43OTkxIDIuNjIyMyA0LjQxNjUgNi43NjI3IDYuOTAwOCA0LjI3ODUgMi40ODQzIDkuMjQ3IDIuNDg0M3oiIHN0eWxlPSJmb250LXZhcmlhdGlvbi1zZXR0aW5nczond2dodCcgNTIyIi8+PHBhdGggZD0ibTc0OS44MyA2Ni42MDJoMTQuNDkydjI1LjUzM2gxNS40NTh2MTEuMzE3aC0xNS40NTh2NDYuNTExaC0xNC40OTJ2LTQ2LjUxMWgtMTAuMjEzdi0xMS4zMTdoMTAuMjEzeiIgc3R5bGU9ImZvbnQtdmFyaWF0aW9uLXNldHRpbmdzOid3Z2h0JyA1MjIiLz48cGF0aCBkPSJtODIyLjI5IDE1MS4zNHEtOS45MzcxIDAtMTcuMjUyLTMuODY0NC03LjE3NjgtNC4wMDI0LTExLjA0MS0xMC45MDMtMy44NjQ0LTcuMDM4OC0zLjg2NDQtMTYuMDEgMC04LjY5NDkgNC4yNzg1LTE1LjU5NiA0LjQxNjUtNy4wMzg4IDExLjczMS0xMS4xNzkgNy40NTI4LTQuMjc4NSAxNi41NjItNC4yNzg1IDExLjczMSAwIDE5LjMyMiA2LjkwMDggNy43Mjg4IDYuNzYyNyAxMC4zNTEgMTkuMDQ2bC00Ny4wNjMgMTYuMTQ4LTMuMzEyNC04LjI4MDkgMzcuNDAyLTEzLjUyNS0zLjAzNjMgMS45MzIycS0xLjUxODItNC4yNzg1LTUuMTA2Ni03LjQ1MjgtMy41ODg0LTMuMzEyNC05LjY2MS0zLjMxMjQtNS4xMDY2IDAtOC45NzEgMi40ODQzLTMuODY0NCAyLjM0NjMtNi4wNzI3IDYuNjI0Ny0yLjIwODIgNC4xNDA0LTIuMjA4MiA5LjY2MSAwIDUuNzk2NiAyLjM0NjMgMTAuMjEzIDIuMzQ2MiA0LjI3ODUgNi40ODY3IDYuNzYyNyA0LjI3ODUgMi4zNDYyIDkuNTIzIDIuMzQ2MiAzLjU4ODQgMCA2LjkwMDgtMS4yNDIxIDMuNDUwNC0xLjM4MDIgNi40ODY3LTMuNDUwNGw2LjYyNDcgMTAuNjI3cS00LjU1NDUgMi44OTgzLTkuOTM3MSA0LjY5MjUtNS4zODI2IDEuNjU2Mi0xMC40ODkgMS42NTYyeiIgc3R5bGU9ImZvbnQtdmFyaWF0aW9uLXNldHRpbmdzOid3Z2h0JyA1MjIiLz48cGF0aCBkPSJtNDA5LjkgMTc0LjQ2cTguMDA0OSAwIDE0LjYzIDIuMDcwMiA2Ljc2MjcgMi4wNzAyIDExLjU5MyA2LjIxMDcgNC44MzA1IDQuMDAyNCA3LjQ1MjggOS45MzcxIDIuNzYwMyA1LjkzNDYgMi43NjAzIDEzLjUyNSAwIDUuOTM0Ni0xLjc5NDIgMTEuODY5LTEuNzk0MiA1Ljc5NjYtNS45MzQ2IDEwLjQ4OS00LjAwMjQgNC42OTI1LTEwLjQ4OSA3LjU5MDgtNi40ODY3IDIuODk4My0xNi4xNDggMi44OTgzaC0xNC42M3YzNS42MDhoLTE1LjA0NHYtMTAwLjJ6bTEuNzk0MiA1MC4wOTlxNS42NTg2IDAgOS4yNDctMS42NTYyIDMuNzI2NC0xLjc5NDIgNS43OTY2LTQuNDE2NSAyLjIwODItMi43NjAzIDMuMDM2My01Ljc5NjYgMC45NjYxMS0zLjE3NDQgMC45NjYxMS01LjkzNDZ0LTAuOTY2MTEtNS43OTY2cS0wLjgyODA5LTMuMDM2My0yLjg5ODMtNS43OTY2LTIuMDcwMi0yLjc2MDMtNS42NTg2LTQuNDE2NS0zLjQ1MDQtMS43OTQyLTguNjk0OS0xLjc5NDJoLTE1LjE4MnYzNS42MDh6bTE2Ljk3NiA4LjgzMyAyNS45NDcgNDEuMjY2aC0xNy41MjhsLTI2LjIyMy00MC44NTJ6IiBzdHlsZT0iZm9udC12YXJpYXRpb24tc2V0dGluZ3M6J3dnaHQnIDUyMiIvPjxwYXRoIGQ9Im00NjMuODYgMjc0LjY2IDQ0LjE2NS0xMDQuMzRoMC44MjgwOWw0NC4xNjUgMTA0LjM0aC0xNy4yNTJsLTMxLjg4MS04MC43MzkgMTAuNzY1LTcuMTc2OC0zNi4wMjIgODcuOTE2em0yNy4wNTEtMzcuNTRoMzUuMzMybDUuMTA2NiAxMi45NzNoLTQ0Ljk5M3oiIHN0eWxlPSJmb250LXZhcmlhdGlvbi1zZXR0aW5nczond2dodCcgNTIyIi8+PHBhdGggZD0ibTY0Ni4yNyAyNjUuNjlxLTIuMDcwMiAxLjc5NDItNS43OTY2IDMuNTg4NC0zLjcyNjQgMS43OTQyLTguNDE4OSAzLjMxMjQtNC42OTI1IDEuMzgwMi05LjUyMyAyLjIwODItNC44MzA1IDAuOTY2MS05LjEwOSAwLjk2NjEtMTIuMTQ1IDAtMjEuOTQ0LTMuNzI2NC05LjY2MS0zLjcyNjQtMTYuNTYyLTEwLjM1MS02Ljc2MjctNi42MjQ3LTEwLjQ4OS0xNS40NTgtMy41ODg0LTguOTcxLTMuNTg4NC0xOS40NiAwLTEyLjk3MyA0LjAwMjQtMjIuOTEgNC4xNDA0LTkuOTM3MSAxMS4xNzktMTYuNyA3LjE3NjgtNi45MDA4IDE2LjU2Mi0xMC4zNTEgOS4zODUtMy40NTA0IDE5LjU5OC0zLjQ1MDQgOC45NzEgMCAxNi41NjIgMi4wNzAyIDcuNzI4OCAyLjA3MDIgMTMuMzg3IDUuMjQ0NmwtNC45Njg1IDEzLjk0cS0yLjc2MDMtMS42NTYyLTYuOTAwOC0zLjAzNjMtNC4wMDI0LTEuMzgwMi04LjI4MDktMi4yMDgyLTQuMjc4NS0wLjk2NjEtNy44NjY4LTAuOTY2MS04LjY5NSAwLTE1LjczNCAyLjYyMjMtNi45MDA4IDIuNDg0My0xMS44NjkgNy4zMTQ4LTQuOTY4NSA0LjgzMDUtNy41OTA4IDExLjczMS0yLjYyMjMgNi43NjI3LTIuNjIyMyAxNS4zMiAwIDguMDA0OSAyLjYyMjMgMTQuNDkyIDIuNzYwMyA2LjQ4NjcgNy43Mjg4IDExLjE3OXQxMS43MzEgNy4zMTQ4cTYuOTAwOCAyLjQ4NDMgMTUuMzIgMi40ODQzIDQuODMwNSAwIDkuMzg1LTAuODI4MDkgNC42OTI1LTAuOTY2MSA4LjAwNDktMi44OTgzdi0xNy45NDJoLTE5LjMyMnYtMTQuNDkyaDM0LjUwNHoiIHN0eWxlPSJmb250LXZhcmlhdGlvbi1zZXR0aW5nczond2dodCcgNTIyIi8+PC9nPjx0ZXh0IHg9IjM2OC4yMTcwNyIgeT0iMTQ5Ljk2MzEyIiBmaWxsPSJub25lIiBmb250LWZhbWlseT0iJ0pvc2VmaW4gU2FucyciIGZvbnQtc2l6ZT0iMTM4LjAycHgiIGxldHRlci1zcGFjaW5nPSIwcHgiIHN0cm9rZS13aWR0aD0iLjk5OTk5IiB3b3JkLXNwYWNpbmc9IjBweCIgc3R5bGU9ImZvbnQtdmFyaWFudC1jYXBzOm5vcm1hbDtmb250LXZhcmlhbnQtZWFzdC1hc2lhbjpub3JtYWw7Zm9udC12YXJpYW50LWxpZ2F0dXJlczpub3JtYWw7Zm9udC12YXJpYW50LW51bWVyaWM6bm9ybWFsO2xpbmUtaGVpZ2h0OjEuMjUiIHhtbDpzcGFjZT0icHJlc2VydmUiPjx0c3BhbiB4PSIzNjguMjE3MDciIHk9IjE0OS45NjMxMiIgc3R5bGU9ImZvbnQtdmFyaWFudC1jYXBzOm5vcm1hbDtmb250LXZhcmlhbnQtZWFzdC1hc2lhbjpub3JtYWw7Zm9udC12YXJpYW50LWxpZ2F0dXJlczpub3JtYWw7Zm9udC12YXJpYW50LW51bWVyaWM6bm9ybWFsO2ZvbnQtdmFyaWF0aW9uLXNldHRpbmdzOid3Z2h0JyA1MjIiPkNsaW1hdGU8L3RzcGFuPjx0c3BhbiB4PSIzNjguMjE3MDciIHk9IjMyMy44Mjk2OCIgZHk9Ii00OS4xNzEwNTkiIHN0eWxlPSJmb250LXZhcmlhbnQtY2Fwczpub3JtYWw7Zm9udC12YXJpYW50LWVhc3QtYXNpYW46bm9ybWFsO2ZvbnQtdmFyaWFudC1saWdhdHVyZXM6bm9ybWFsO2ZvbnQtdmFyaWFudC1udW1lcmljOm5vcm1hbDtmb250LXZhcmlhdGlvbi1zZXR0aW5nczond2dodCcgNTIyIj5SQUc8L3RzcGFuPjwvdGV4dD48L3N2Zz4K' style='height: 5em; margin: 1em;'></div>"
    )
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

            with gr.Column(scale=4):
                chat_header = gr.Markdown("## Chat", height=100)
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
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
                headers=["Date", "Question", "Answer", "PDF", "DOCX"],
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
            search_input = gr.Textbox(placeholder='Search documents. eg. carbontracker or @title:"annual report 2022 shell"', show_label=False)
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
        )
        for bot_message, questions, answers in bot_messages:
            chat_history.append([None, bot_message])
            yield chat_history, chat_history, questions, answers, chat_id, *download_latest_answer(
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
            chat_id_state,
            rag_filter_textbox,
            improve_question_checkbox,
            do_rerank_checkbox,
            do_crawl_checkbox,
            max_search_queries_textbox,
            do_add_additional_metadata_checkbox,
            language_dropdown,
            do_initial_generation_checkbox,
        ],
        [
            chatbot,
            chat_state,
            questions_state,
            answers_state,
            chat_id_state,
            download_word_button,
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

    def filter_documents(search_query):
        from query_data import query_source_documents

        if search_query is None:
            search_query = ""
        if len(search_query) < 2:
            return gr.Dataset(samples=[])
        if bool(re.search(r"@.+:.+", search_query)) is False:
            # Turn into a search query if not a ft.search-style query
            search_query = f"@source:*{search_query}*"

        search_results = query_source_documents(
            db, search_query, print_output=False, fields=["source"], limit=30
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
        fn=lambda x: filter_documents(x),
        inputs=[rag_filter_textbox],
        outputs=[doc_sources],
        queue=False,
    )

    ## Tab 2: Query history
    def get_query_history(query_filter):
        import pandas as pd
        import msgspec
        import datetime
        from helpers import get_previous_queries

        if (query_filter is None) or (query_filter == ""):
            query_filter = "*"

        # Get dataframe of previous queries
        df = get_previous_queries(r, query_filter=query_filter, limit=100).set_index(
            "qa_id"
        )[["date_added_ts", "question", "answer", "pdf_uri", "docx_uri"]]
        # Deal with missing PDF and DOCX URIs
        missing_pdfs = df.loc[lambda df: df.isnull().any(axis=1)].index.tolist()
        for qa_id in missing_pdfs:
            df.loc[qa_id, "pdf_uri"], df.loc[qa_id, "docx_uri"] = render_qa_pdfs(qa_id)
        # Sanitize URLs
        df.pdf_uri = df.pdf_uri.apply(sanitize_url)
        df.docx_uri = df.docx_uri.apply(sanitize_url)

        df.pdf_uri = (
            "<a target='_blank' href='"
            + df.pdf_uri.astype(str).fillna("")
            + "'><img src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDI0IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPHBhdGggZD0iTTUgNGgxMHY0aDR2MTJINXpNMy45OTkgMkEuOTk1Ljk5NSAwIDAgMCAzIDIuOTkydjE4LjAxNmExIDEgMCAwIDAgLjk5My45OTJoMTYuMDE0QTEgMSAwIDAgMCAyMSAyMC45OTJWN2wtNS01em02LjUgNS41YzAgMS41NzctLjQ1NSAzLjQzNy0xLjIyNCA1LjE1M2MtLjc3MiAxLjcyMy0xLjgxNCAzLjE5Ny0yLjkgNC4wNjZsMS4xOCAxLjYxM2MyLjkyNy0xLjk1MiA2LjE2OC0zLjI5IDkuMzA0LTIuODQybC40NTctMS45MzlDMTQuNjQ0IDEyLjY2MSAxMi41IDkuOTkgMTIuNSA3LjV6bS42IDUuOTcyYy4yNjgtLjU5Ny41MDUtMS4yMTYuNzA1LTEuODQzYTkuNyA5LjcgMCAwIDAgMS43MDYgMS45NjZjLS45ODIuMTc2LTEuOTQ0LjQ2NS0yLjg3NS44MzNxLjI0OC0uNDcxLjQ2NS0uOTU2IiBmaWxsPSIjZmZmIiBzdHJva2U9IiMwMDAiIHN0cm9rZS13aWR0aD0iLjIiLz4KPC9zdmc+Cg==' alt='PDF'></img></a>"
        )
        df.docx_uri = (
            "<a target='_blank' href='"
            + df.docx_uri.astype(str).fillna("")
            + "'><img src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDI0IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPHBhdGggZD0iTTE2IDh2OGgtMmwtMi0ybC0yIDJIOFY4aDJ2NWwyLTJsMiAyVjhoMVY0SDV2MTZoMTRWOHpNMyAyLjk5MkMzIDIuNDQ0IDMuNDQ3IDIgMy45OTkgMkgxNmw1IDV2MTMuOTkzQTEgMSAwIDAgMSAyMC4wMDcgMjJIMy45OTNBMSAxIDAgMCAxIDMgMjEuMDA4eiIgZmlsbD0iI2ZmZiIgc3Ryb2tlPSIjMDAwIiBzdHJva2Utd2lkdGg9Ii4yIi8+Cjwvc3ZnPg==' alt='DOCX'></img></a>"
        )

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
    def search_documents(search_query):
        from query_data import query_source_documents

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
            limit=100
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
        inputs=[url_input],
        outputs=[selected_source],
        queue=False,
    )
    url_input.submit(
        fn=add_document,
        inputs=[url_input],
        outputs=[selected_source],
        queue=False,
    )

    # Upload new document
    from tools import upload_documents

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


demo.queue(default_concurrency_limit=None)
demo.launch(inbrowser=False, show_api=False, max_threads=80)
