import gradio as gr
from loader import translate, load_model

# load model once at startup before any requests arrive
load_model()

def translate_input(text):
    if not text.strip():
        return "Please enter some text to translate."
    try:
        return translate(text)
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Blocks gives full layout control vs the simpler gr.Interface
with gr.Blocks(title="English → Marathi Translator") as demo:
    gr.Markdown("# English → Marathi Translator")
    gr.Markdown("Built with a Transformer trained from scratch in PyTorch.")

    with gr.Row():
        input_box  = gr.Textbox(label="English", placeholder="Enter English text...")
        output_box = gr.Textbox(label="Marathi",  interactive=False)

    translate_btn = gr.Button("Translate", variant="primary")
    translate_btn.click(fn=translate_input, inputs=input_box, outputs=output_box)

    gr.Examples(
        examples=["Hello, how are you?", "What is your name?", "I love learning."],
        inputs=input_box
    )

if __name__ == "__main__":
    # share=True generates a public link valid for 72 hours
    demo.launch(share=True)