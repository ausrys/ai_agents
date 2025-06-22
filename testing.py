from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# --- 1. Define state ---

torch.cuda.empty_cache()
torch.cuda.ipc_collect()


class AgentState(TypedDict):
    input: str
    output: str


# --- 2. Define tools ---
text_gen = pipeline("text2text-generation", model="google/flan-t5-xl",
                    device=-1)
image_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5")
image_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Define nodes ---


def classify_task(state: AgentState) -> AgentState:
    prompt = state["input"]
    kind = "image" if any(w in prompt.lower()
                          for w in ["draw", "picture", "visualize"]) else "text"
    return {"input": prompt, "output": kind}  # tags the kind


def handle_text(state: AgentState) -> AgentState:
    prompt = state['input']
    response = text_gen(prompt, max_length=150, do_sample=True)[
        0]['generated_text']
    return {"input": prompt, "output": response}


def handle_image(state: AgentState) -> AgentState:
    img = image_pipe(state["input"]).images[0]
    path = "generated.png"
    img.save(path)
    return {"input": state["input"], "output": f"Saved image at {path}"}


# --- 4. Build graph ---
builder = StateGraph(AgentState)
builder.add_node("classify", classify_task)
builder.add_node("text", handle_text)
builder.add_node("image", handle_image)

# flow: START -> classify
builder.add_edge(START, "classify")

# conditional: depending on classify.output go to text or image


def route(state: AgentState) -> str:
    return state["output"]  # either "text" or "image"


builder.add_conditional_edges("classify", route, {
    "text": "text",
    "image": "image"
})

# terminal
builder.add_edge("text", END)
builder.add_edge("image", END)

graph = builder.compile()


res1 = graph.invoke(
    {"input": "What is the Python programming language used for? Explain simply."})
print(res1["output"])  # text answer

res2 = graph.invoke(
    {"input": "Create a picture of futuristic city", "output": ""})
print(res2["output"])  # image saved message
