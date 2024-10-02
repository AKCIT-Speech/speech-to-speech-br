import gradio as gr
import os

def get_chatbot_response(x):
    os.rename(x, x + '.wav')
    return [((x + '.wav',), "Your voice sounds nice!")]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    mic = gr.Audio(sources="microphone", type="filepath")
    mic.change(get_chatbot_response, mic, chatbot)


demo.launch()
