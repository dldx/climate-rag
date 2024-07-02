import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from query_data import run_query
from tools import get_vector_store

db = get_vector_store()


def climate_chat(
    message,
    history,
    questions,
    rag_filter,
    improve_question,
    do_rerank,
    language,
    initial_generation,
):
    happy_with_answer = True
    getting_feedback = False
    if len(history) > 1:
        getting_feedback = history[-2][1] == "Are you happy with the answer? (y/n)"
        if getting_feedback:
            if message.lower() == "y":
                happy_with_answer = True
            elif message.lower() == "n":
                happy_with_answer = False
            else:
                yield "Please enter 'y' or 'n'", questions
        if happy_with_answer:
            yield "Great! I'm glad I could help. What else would you like to know?", questions

    if getting_feedback:
        message = questions[-1]
    else:
        questions.append(message)
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
            if (key == "improve_question") and improve_question:
                yield f"""Better question: {value["question"]}

                Better question (en): {value["question_en"]}""", questions
            elif key == "retrieve_from_database":
                yield f"""Search queries: {value["search_prompts"]}

                Retrieved from database: {[doc.metadata["id"] for doc in value["documents"]]}""", questions
            elif key == "web_search_node":
                yield f"""Search query: {value["search_query"]}

                Search query (en): {value["search_query_en"]}""", questions
            elif key == "rerank_documents":
                yield f"""Reranked documents: {[doc.metadata["id"] for doc in value["documents"]]}""", questions

            elif key == "generate":
                yield (
                    f"""# {value["initial_question"]}\n\n"""
                    + value["generation"]
                    + "\n\n**Sources:**\n\n"
                    + "\n\n".join(
                        set(
                            [
                                (
                                    " * " + doc.metadata["source"]
                                    if "source" in doc.metadata.keys()
                                    else ""
                                )
                                for doc in value["documents"]
                            ]
                        )
                    )
                ), questions
            elif key == "add_urls_to_database":
                yield f"""Added new pages to database""", questions
            elif key == "ask_user_for_feedback":
                yield f"""Are you happy with the answer? (y/n)""", questions
            else:
                yield str(value), questions


with gr.Blocks(
    fill_height=True,
    css="""
.h-80 {
    height: 80vh;
}

""",
) as demo:
    # Define how to store state
    chat_state = gr.State([])
    questions_state = gr.State([])

    gr.Markdown("# Climate RAG")
    with gr.Tab("Chat"):
        # Add a header
        with gr.Row(elem_classes=["h-80"]):
            with gr.Column(variant="panel"):
                gr.Markdown("## Documents retrieved")
                datasets = gr.Dataset(components=[])
            with gr.Column(scale=4):
                gr.Markdown("## Chat")
                chatbot = gr.Chatbot(scale=4, show_label=False)
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question. e.g. What does the Shell 2023 annual report say about climate change?",
                        show_label=False,
                    )
                with gr.Row():
                    improve_question_checkbox = gr.Checkbox(
                        value=True, label="Auto-improve question?"
                    )
                    do_initial_generation_checkbox = gr.Checkbox(
                        value=True, label="Generate answer before web search?"
                    )
                    do_rerank_checkbox = gr.Checkbox(
                        value=True, label="Rerank documents?"
                    )
                    language_dropdown = gr.Dropdown(
                        choices=[("English", "en"), ("Chinese", "zh"), ("Italian", "it"), ("Spanish", "es"), ("German", "de"), ("Vietnamese", "vi")],
                        label="Language",
                        value="en",
                        filterable=True,
                    )
                    rag_filter_textbox = gr.Textbox(
                        placeholder="Document url filter. eg. carbontracker.org",
                        show_label=False,
                    )

                with gr.Row():
                    stop_button = gr.Button(value="Stop", variant="stop")
                    clear = gr.ClearButton([msg, chatbot])
    with gr.Tab("Documents"):
        with gr.Row():
            gr.Markdown("## Add new documents")
        with gr.Row():
            new_file = gr.File(label="Upload documents", file_types=["pdf", "PDF"])
            with gr.Column():
                url_input = gr.Textbox(placeholder="Enter a URL", show_label=False)
                add_button = gr.Button(value="Add")
        with gr.Row():
            gr.Markdown("## Search through existing documents")
        # Search through documents in the vector database
        with gr.Row():
            search_input = gr.Textbox(placeholder="Search documents", show_label=False)
            search_button = gr.Button(value="Search")
        with gr.Row():
            search_results_display = gr.Dataset(components=[gr.Textbox(visible=False), gr.Markdown(visible=False), gr.Number(visible=False)],
                                                label="Documents retrieved",
                                                headers=["Source", "Page contents", "Number of tokens"])

    ### Define the logic
    ## Tab 1: Chat
    def update_questions(questions):
        print("Updating questions")
        return [gr.Checkbox(label=q) for q in questions]

    def user(user_message, history):
        return "", history + [[user_message, None]], history + [[user_message, None]]

    def bot(
        chat_history,
        questions,
        rag_filter,
        improve_question,
        do_rerank,
        language,
        initial_generation,
    ):
        message = chat_history[-1][0]
        bot_messages = climate_chat(
            message=message,
            history=chat_history,
            questions=questions,
            rag_filter=rag_filter,
            improve_question=improve_question,
            do_rerank=do_rerank,
            language=language,
            initial_generation=initial_generation,
        )
        chat_history[-1][0]
        for bot_message, questions in bot_messages:
            chat_history.append([None, bot_message])
            yield chat_history, chat_history, questions

    converse_event = msg.submit(
        fn=user,
        inputs=[msg, chat_state],
        outputs=[msg, chatbot, chat_state],
        queue=False,
    ).then(
        bot,
        [
            chatbot,
            questions_state,
            rag_filter_textbox,
            improve_question_checkbox,
            do_rerank_checkbox,
            language_dropdown,
            do_initial_generation_checkbox,
        ],
        [chatbot, chat_state, questions_state],
    )

    stop_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[converse_event],
        queue=False,
    )
    questions_state.change(
        fn=update_questions, inputs=[questions_state], outputs=[datasets], queue=False
    )
    clear.click(
        fn=lambda: None, inputs=None, outputs=[chatbot, chat_state], queue=False
    )

    ## Tab 2: Documents
    def search_documents(search_query):
        from query_data import query_source_documents_by_metadata
        search_results = query_source_documents_by_metadata(db, "source", search_query, print=False)
        return gr.Dataset(samples=search_results.to_numpy().tolist())

    search_button.click(
        fn=search_documents,
        inputs=[search_input],
        outputs=[search_results_display],
        queue=False,
    )

    # Add new documents
    def add_document(url):
        from tools import add_urls_to_db_firecrawl
        add_urls_to_db_firecrawl([url], db)
        return url

    add_button.click(
        fn=add_document,
        inputs=[url_input],
        outputs=[search_input],
        queue=False,
    ).then(fn=lambda: None, inputs=None, outputs=[url_input], queue=False).then(
        search_documents,
        [search_input],
        [search_results_display],
    )

    # Upload new document
    def upload_document(file):
        import requests
        from tools import add_urls_to_db_firecrawl
        filename = file.split("/")[-1]
        response = requests.post(url='https://tmpfiles.org/api/v1/upload', files={"file": open(file, 'rb')})
        # Store filename with URL
        dl_url = "https://tmpfiles.org/dl/" + response.json()["data"]["url"].replace("https://tmpfiles.org", "") + "#" + filename
        add_urls_to_db_firecrawl([dl_url], db)
        return dl_url

    new_file.upload(
        fn=upload_document,
        inputs=[new_file],
        outputs=[search_input],
        queue=False,
    ).then(
        search_documents,
        [search_input],
        [search_results_display],
    )

demo.queue()
demo.launch(inbrowser=True, show_api=False)
