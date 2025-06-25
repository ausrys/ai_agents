from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from newspaper import Article
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
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# --- 3. Define nodes ---


def classify_task(state: AgentState) -> AgentState:
    prompt = state["input"].lower().strip()

    if prompt.startswith("http://") or prompt.startswith("https://"):
        return {"input": state["input"], "output": "web"}

    elif any(w in prompt for w in ["draw", "image", "picture", "visualize"]):
        return {"input": state["input"], "output": "image"}

    elif any(w in prompt for w in ["summarize", "summary", "shorten this"]):
        return {"input": state["input"], "output": "summarize"}

    else:
        return {"input": state["input"], "output": "text"}


def handle_summary(state: AgentState) -> AgentState:
    prompt = state['input']
    summary = summarizer(prompt, max_length=100, min_length=30, do_sample=False)[
        0]['summary_text']
    return {"input": prompt, "output": summary}


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


def handle_webpage(state: AgentState) -> AgentState:
    url = state["input"].strip()

    try:
        article = Article(url)
        article.download()
        article.parse()
        content = article.text[:3000]  # limit input size for summarizer
    except Exception as e:
        return {"input": url, "output": f"Error parsing web page: {str(e)}"}

    # Optional: summarize it
    summary = summarizer(content, max_length=100, min_length=30, do_sample=False)[
        0]['summary_text']
    return {"input": url, "output": summary}


# --- 4. Build graph ---
builder = StateGraph(AgentState)
builder.add_node("classify", classify_task)
builder.add_node("text", handle_text)
builder.add_node("image", handle_image)
builder.add_node("summarize", handle_summary)
builder.add_node("web", handle_webpage)
# flow: START -> classify
builder.add_edge(START, "classify")

# conditional: depending on classify.output go to text or image


def route(state: AgentState) -> str:
    return state["output"]  # either "text" or "image"


builder.add_conditional_edges("classify", route, {
    "text": "text",
    "image": "image",
    "summarize": "summarize",
    "web": "web"
})

# terminal
builder.add_edge("text", END)
builder.add_edge("image", END)
builder.add_edge("summarize", END)
builder.add_edge("web", END)
graph = builder.compile()


res1 = graph.invoke(
    {"input": "What is the Python programming language used for? Explain simply."})
print(res1["output"])  # text answer

res2 = graph.invoke(
    {"input": "Create a picture of futuristic city", "output": ""})
print(res2["output"])  # image saved message

long_text = """
Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. This includes capabilities like learning, problem-solving, and decision-making. AI is a broad field encompassing various technologies, including machine learning and deep learning. 
Here's a more detailed explanation:
What AI does: AI systems are designed to perform tasks that typically require human intelligence. This can involve: 
Learning: Adapting to new information and improving performance over time.
Reasoning: Drawing conclusions and making logical inferences.
Problem-solving: Finding solutions to complex challenges.
Decision-making: Choosing the best course of action based on available data.
Perception: Enabling machines to "see" and understand the world around them.
Language understanding: Interpreting and responding to human language.
"""

result_summary = graph.invoke(
    {"input": f"Summarize this: {long_text}", "output": ""})
print(result_summary["output"])


result_parser = graph.invoke(
    {"input": "https://en.wikipedia.org/wiki/Python_(programming_language)", "output": ""})
print(result_parser["output"])
