from langgraph.graph import StateGraph, START, END
from paddleocr import PaddleOCR
import pandas as pd
import requests
from typing import TypedDict
import os
from bs4 import BeautifulSoup
from PIL import Image
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


class AgentState(TypedDict):
    input: str
    output: str
    files: list[str]


ocr_model = PaddleOCR(use_angle_cls=True, lang='en')


def handle_table_extraction(state: AgentState) -> AgentState:
    prompt = state["input"]
    output_type = state.get("format", "csv").lower()

    if prompt.startswith("http"):
        try:
            tables = pd.read_html(prompt)
            output_files = []

            for i, table in enumerate(tables):
                filename = f"table_{i}.{output_type}"
                if output_type == "csv":
                    table.to_csv(filename, index=False)
                else:
                    table.to_excel(filename, index=False)
                output_files.append(filename)

            return {
                "input": prompt,
                "output": f"{len(output_files)} table(s) extracted.",
                "files": output_files
            }

        except Exception as e:
            return {
                "input": prompt,
                "output": f"Failed to extract HTML tables: {str(e)}"
            }

    elif os.path.isfile(prompt) and prompt.lower().endswith((".png", ".jpg", ".jpeg")):
        result = ocr_model.ocr(prompt)
        table_data = [line[1][0] for line in result[0]]
        df = pd.DataFrame([row.split() for row in table_data])

        filename = f"ocr_table.{output_type}"
        if output_type == "csv":
            df.to_csv(filename, index=False)
        else:
            df.to_excel(filename, index=False)

        return {
            "input": prompt,
            "output": f"OCR table saved.",
            "files": [filename]
        }

    else:
        return {
            "input": prompt,
            "output": "Invalid input. Provide a Wikipedia URL or image path."
        }


def classify_task(state: AgentState) -> AgentState:
    prompt = state["input"].lower().strip()

    if prompt.startswith("http") or prompt.endswith((".csv", ".xlsx", ".xls", ".html")):
        return {"input": state["input"], "output": "table"}

    if os.path.isfile(prompt) and prompt.lower().endswith((".png", ".jpg", ".jpeg")):
        return {"input": state["input"], "output": "table"}

    return {"input": state["input"], "output": "unsupported"}


def route(state: AgentState) -> str:
    return state["output"]  # either "text" or "image"


builder = StateGraph(AgentState)
builder.add_node("classify", classify_task)
builder.add_node("table", handle_table_extraction)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, {
    "table": "table",
    "unsupported": END
})
builder.add_edge("table", END)
graph = builder.compile()

res = graph.invoke({
    "input": "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)",
    "format": "xlsx"
})

print("Output:", res["output"])
print("Files:", res.get("files"))

res = graph.invoke({
    "input": "table.png",
    "format": "csv"
})
print("Output:", res["output"])
print("Files:", res.get("files"))
