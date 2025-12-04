import gradio as gr
from dotenv import load_dotenv
from implementation.answer import answer_question


load_dotenv(override=True)

def on_submit(question, history):
    history.append({"role": "user", "content": question})
    answer, _ = answer_question(question, history)
    history.append({"role": "assistant", "content": answer})
    return history, history, ""

with gr.Blocks() as ui:
    gr.Markdown("# Sadegh Modern Resume\nAsk me anything about Sadegh Mahmoud Abadi!")

    chatbot = gr.Chatbot(label="Conversation")
    question_box = gr.Textbox(label="Ask a question")
    history_state = gr.State([])

    submit_btn = gr.Button("Ask")

    submit_btn.click(
        fn=on_submit,
        inputs=[question_box, history_state],
        outputs=[chatbot, history_state, question_box]
    )

    question_box.submit(
        fn=on_submit,
        inputs=[question_box, history_state],
        outputs=[chatbot, history_state, question_box]
    )

ui.launch()
