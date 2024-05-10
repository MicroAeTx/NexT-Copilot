#你可以自己定义的部分(模型文件夹的绝对路径) #Your customizable section(absolute path to model folder)
model_name_path = "F:\INT8_compressed_weights"


from config import SUPPORTED_LLM_MODELS
import ipywidgets as widgets

model_language = input("#Chinese 你想要使用什么语言呢，English or Chinese？#English What language do you want to use? English or Chinese?")

model_language
model_ids = list(SUPPORTED_LLM_MODELS[model_language])

model_id = "minicpm-2b-dpo"
model_configuration = SUPPORTED_LLM_MODELS[model_language][model_id]
print(f"Selected model {model_id}")
from transformers import AutoConfig
from optimum.intel.openvino import OVModelForCausalLM
from pathlib import Path
import torch
from converter import register_configs

register_configs()

prepare_int8_model = widgets.Checkbox(
    value=False,
    description="Prepare INT8 model",
    disabled=False,
)


int8_model_dir = Path(model_id) / "INT8_compressed_weights"
pt_model_name = model_id.split("-")[0]
int8_weights = int8_model_dir / "openvino_model.bin"

device = "GPU"

from ov_llm_model import model_classes

available_models = []
if int8_model_dir.exists():
    available_models.append("INT8")

model_to_run = "int8"

from transformers import AutoTokenizer

model_dir = int8_model_dir
print(f"Loading model from {model_dir}")

ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

model_name = model_name_path
class_key = model_id.split("-")[0]
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model_class = (
    OVModelForCausalLM
    if not model_configuration["remote"]
    else model_classes[class_key]
)
ov_model = model_class.from_pretrained(
    model_dir,
    device=device,
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)

from threading import Event, Thread
from uuid import uuid4
from typing import List, Tuple
import gradio as gr
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


model_name = model_configuration["model_id"]
start_message = model_configuration["start_message"]
history_template = model_configuration.get("history_template")
current_message_template = model_configuration.get("current_message_template")
stop_tokens = model_configuration.get("stop_tokens")
tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})

chinese_examples = [
    ["你好!"],
    ["你是谁?"],
    ["请介绍一下上海"],
    ["请介绍一下英特尔公司"],
    ["晚上睡不着怎么办？"],
    ["给我讲一个年轻人奋斗创业最终取得成功的故事。"],
    ["给这个故事起一个标题。"],
]

english_examples = [
    ["Hello there! How are you doing?"],
    ["What is OpenVINO?"],
    ["Who are you?"],
    ["Can you explain to me briefly what is Python programming language?"],
    ["Explain the plot of Cinderella in a sentence."],
    ["What are some common mistakes to avoid when writing code?"],
    [
        "Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“"
    ],
]

examples = (
    chinese_examples
    if (model_language == "Chinese")
    else english_examples
    if (model_language == "English")
    else english_examples
)

max_new_tokens = 4096


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = tok.convert_tokens_to_ids(stop_tokens)

    stop_tokens = [StopOnTokens(stop_tokens)]


def default_partial_text_processor(partial_text: str, new_text: str):
    """
    helper for updating partially generated answer, used by default

    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string

    """
    partial_text += new_text
    return partial_text


text_processor = model_configuration.get(
    "partial_text_processor", default_partial_text_processor
)


def convert_history_to_token(history: List[Tuple[str, str]]):
    """
    function for conversion history stored as list pairs of user and assistant messages to tokens according to model expected conversation template
    Params:
      history: dialogue history
    Returns:
      history in token format
    """
    if history_template is None:
        messages = [{"role": "system", "content": start_message}]
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})
                
        input_token = tok.apply_chat_template(messages,
                                              add_generation_prompt=True,
                                              tokenize=True,
                                              return_tensors="pt")
    else:
        text = start_message + "".join(
            [
                "".join(
                    [
                        history_template.format(
                            num=round, user=item[0], assistant=item[1]
                        )
                    ]
                )
                for round, item in enumerate(history[:-1])
            ]
        )
        text += "".join(
            [
                "".join(
                    [
                        current_message_template.format(
                            num=len(history) + 1,
                            user=history[-1][0],
                            assistant=history[-1][1],
                        )
                    ]
                )
            ]
        )
        input_token = tok(text, return_tensors="pt", **tokenizer_kwargs).input_ids
    return input_token


def user(message, history):
    """
    callback function for updating user messages in interface on submit button click

    Params:
      message: current message
      history: conversation history
    Returns:
      None
    """
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
    """
    callback function for running chatbot on submit button click

    Params:
      history: conversation history
      temperature:  parameter for control the level of creativity in AI-generated text.
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
      top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
      top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
      repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
      conversation_id: unique conversation identifier.

    """

    # Construct the input message string for the model by concatenating the current system message and conversation history
    # Tokenize the messages string
    input_ids = convert_history_to_token(history)
    if input_ids.shape[1] > 4096:
        history = [history[-1]]
        input_ids = convert_history_to_token(history)
    streamer = TextIteratorStreamer(
        tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    if stop_tokens is not None:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
            stop_tokens)

    stream_complete = Event()

    def generate_and_signal_complete():
        """
        genration function for single thread
        """
        global start_time
        ov_model.generate(**generate_kwargs)
        stream_complete.set()

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text = text_processor(partial_text, new_text)
        history[-1][1] = partial_text
        yield history


def request_cancel():
    ov_model.request.cancel()


def get_uuid():
    """
    universal unique identifier for thread
    """
    return str(uuid4())


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    conversation_id = gr.State(get_uuid)
    gr.Markdown(
        f"""<h1><center>NexT-Copilot {model_id} Chatbot</center></h1>""")
    chatbot = gr.Chatbot(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
                container=False,
            )
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    with gr.Row():
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.1,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                with gr.Column():
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=1.0,
                            minimum=0.0,
                            maximum=1,
                            step=0.01,
                            interactive=True,
                            info=(
                                "Sample from the smallest possible set of tokens whose cumulative probability "
                                "exceeds top_p. Set to 1 to disable and sample from all tokens."
                            ),
                        )
                with gr.Column():
                    with gr.Row():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=50,
                            minimum=0.0,
                            maximum=200,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                        )
                with gr.Column():
                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            value=1.1,
                            minimum=1.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Penalize repetition — 1.0 to disable.",
                        )
    gr.Examples(
        examples, inputs=msg, label="Click on any example and press the 'Submit' button"
    )

    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )
    stop.click(
        fn=request_cancel,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)

# if you are launching remotely, specify server_name and server_port
#  demo.launch(server_name='your server name', server_port='server port in int')
# if you have any issue to launch on your platform, you can pass share=True to launch method:
# demo.launch(share=True)
# it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
demo.launch()
