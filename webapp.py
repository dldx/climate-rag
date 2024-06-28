import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from query_data import run_query
from tools import get_vector_store

db = get_vector_store()

def climate_chat(message, history, questions):
    happy_with_answer = True
    getting_feedback = False
    if len(history) > 1:
        getting_feedback = history[-2][1] == "Are you happy with the answer? (y/n)"
        if getting_feedback:
            happy_with_answer = message.lower() == "y"
        if happy_with_answer:
            yield "Great! I'm glad I could help. What else would you like to know?", questions


    if getting_feedback:
        message = questions[-1]
    else:
        questions.append(message)
    if (getting_feedback and not happy_with_answer) or not getting_feedback:
        for key, value in run_query(message, db, llm="gpt-4o", mode="gui",
                                    history=history,
                                    initial_generation=happy_with_answer):
            if key == "improve_question":
                yield f"""Better question: {value["question"]}

                Better question (en): {value["question_en"]}""", questions
            elif key == "formulate_query":
                yield value["search_prompts"], questions
            elif key == "generate_search_query":
                yield f"""Search query: {value["search_query"]}

                Search query (en): {value["search_query_en"]}""", questions
            elif key == "retrieve_from_database":
                yield f"""Retrieved from database: {[doc.metadata["id"] for doc in value["documents"]]}""", questions
            elif key == "web_search_node":
                yield f"""Search query: {value["search_query"]}

                Search query (en): {value["search_query_en"]}""", questions
            elif key == "rerank_documents":
                yield f"""Reranked documents: {[doc.metadata["id"] for doc in value["documents"]]}""", questions

            elif key == "generate":
                yield (f"""# {value["question"]}\n\n"""
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
                            )), questions
            elif key == "ask_user_for_feedback":
                yield f"""Are you happy with the answer? (y/n)""", questions
            else:
                yield str(value), questions


with gr.Blocks(fill_height=True) as demo:
    # Define how to store state
    chat_state = gr.State([])
    questions_state = gr.State([])

    gr.Markdown("# Climate RAG")
    with gr.Tab("Chat"):
        # Add a header
        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("## Documents retrieved")
                datasets = gr.Dataset(components=[])
            with gr.Column(scale=4):
                gr.Markdown("## Chat")
                chatbot = gr.Chatbot(scale=2, show_label=False)
                msg = gr.Textbox(placeholder="Ask a question. e.g. What does the Shell 2023 annual report say about climate change?")
                clear = gr.ClearButton([msg, chatbot])
    with gr.Tab("Documents"):
        # Search through documents in the vector database
        with gr.Row():
            search = gr.Textbox(placeholder="Search documents", show_label=False)
            search_button = gr.Button(value="Search")
        with gr.Row():
            search_results = gr.Dataframe()
            search_results_display = gr.Dataset(components=[])



    def update_questions(questions):
        print("Updating questions")
        return [gr.Checkbox(label=q) for q in questions]


    def user(user_message, history):
        return "", history + [[user_message, None]], history + [[user_message, None]]

    def bot(chat_history, questions):
        message = chat_history[-1][0]
        bot_messages = climate_chat(message, chat_history, questions)
        chat_history[-1][0]
        for bot_message, questions in bot_messages:
            chat_history.append([None, bot_message])
            yield chat_history, chat_history, questions

    msg.submit(fn=user, inputs=[msg, chat_state], outputs=[msg, chatbot, chat_state], queue=False).then(
        bot, [chatbot, questions_state], [chatbot, chat_state, questions_state]
    )
    questions_state.change(fn=update_questions, inputs=[questions_state], outputs=[datasets], queue=False)
    clear.click(fn=lambda: None, inputs=None, outputs=[chatbot, chat_state], queue=False)

demo.queue()
demo.launch()