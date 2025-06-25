# 🧠 Multimodal AI Agent with LangGraph, Smol-Agent, and Hugging Face

This project implements a smart AI agent using [LangGraph](https://huggingface.co/learn/agents-course/en/unit2/langgraph/first_graph), [Smol-Agent](https://huggingface.co/smol-ai/smol-agent), and [Hugging Face Transformers/Diffusers] to process various types of user inputs. The agent can:

✅ Answer questions using a large language model  
✅ Generate images from text prompts  
✅ Summarize long text inputs  
✅ Parse and summarize content from live web pages (URLs)

---

## ✨ Features

| Input Type             | Tool Used                  | Output                          |
| ---------------------- | -------------------------- | ------------------------------- |
| Free-form question     | `flan-t5-xl` or other LLM  | Informative answer              |
| Image request          | `stable-diffusion-v1-5`    | Generated image (saved locally) |
| Long text to summarize | `facebook/bart-large-cnn`  | Concise summary                 |
| Web page URL           | `newspaper3k` + summarizer | Summary of web page content     |

---

## 🛠️ Stack

- **LangGraph** – for defining conditional agent flows
- **Transformers** – text generation (`flan-t5-xl`)
- **Diffusers** – image generation (`stable-diffusion-v1-5`)
- **newspaper3k** – web page parsing
- **PyTorch** – GPU support for heavy models

---

## 🚀 Setup Instructions

### 1. Clone and create a Python environment

```bash
git clone https://github.com/your-username/ai-agent-langgraph
cd ai-agent-langgraph
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 💡 How It Works

The agent is built as a graph with these nodes:

classify: Determines what kind of input the user gave (text, image, summarize, or web).

text: Uses a transformer model to generate an answer.

image: Uses Stable Diffusion to generate an image.

summarize: Summarizes a long block of text.

web: Downloads and extracts content from a live web page, then summarizes it.

LangGraph manages routing between these steps based on your input type.

## 🧪 Example Inputs

```python
graph.invoke({"input": "Explain what is machine learning", "output": ""})
graph.invoke({"input": "Draw a cat on the moon", "output": ""})
graph.invoke({"input": "Summarize this: <long paragraph>", "output": ""})
graph.invoke({"input": "https://en.wikipedia.org/wiki/Python_(programming_language)", "output": ""})
```
## 🧾 Table Extraction Agent (OCR + HTML)


This agent extracts tables from either Wikipedia URLs or image files (JPEG/PNG) and saves them as .csv or .xlsx files, based on user input.

✅ Supported Inputs:
Wikipedia URLs containing HTML tables

Image files (screenshots of tables) using OCR

🧠 How It Works:
If the input is a URL:

Uses pandas.read_html() to extract all HTML tables from the page.

Saves each table to a separate file (e.g., table_0.csv, table_1.csv).

If the input is an image path:

Uses PaddleOCR to detect and recognize table text.

Constructs a rough table by splitting text into rows/columns.

Saves it as ocr_table.csv or ocr_table.xlsx.

## 🧠 Credits


Hugging Face LangGraph

Transformers & Diffusers

newspaper3k

