import os
import gradio as gr
from scripts.insert_search import image_search

def main():
    gr.close_all()
    app = gr.Blocks(theme='default', title="image",
                css=".gradio-container, .gradio-container button {background-color: #009FCC} "
                    "footer {visibility: hidden}")
    with app:
        with gr.Tabs():
            with gr.TabItem("image search"):
                with gr.Row():
                    with gr.Column():
                        text = gr.TextArea(label="Text",
                                            placeholder="description",
                                            value="",)
                        img_input=gr.inputs.Image(type="pil", label="Image")
                        btn = gr.Button(label="Search", button_type="submit")

                    with gr.Column():
                        with gr.Row():
                            output_images = [gr.outputs.Image(type="pil", label=None) for _ in range(6)]
                            

                    btn.click(image_search, inputs=[text,img_input], outputs=output_images, show_progress=True)

    # ip_addr = net_helper.get_host_ip()
    ip_addr = '0.0.0.0'
    app.queue(concurrency_count=3).launch(show_api=False, share=True, server_name=ip_addr, server_port=6006)

if __name__ == '__main__':
    main()
