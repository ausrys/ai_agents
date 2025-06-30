import os
import shutil
from git import Repo
from transformers import pipeline

# Load a lightweight model
test_gen = pipeline(
    "text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")


def clone_repo(url: str, dest: str = "repo"):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    Repo.clone_from(url, dest)
    return dest


def get_python_files(repo_path: str):
    py_files = []
    for root, _, files in os.walk(repo_path):
        for f in files:
            if f.endswith(".py") and "test" not in f.lower():
                py_files.append(os.path.join(root, f))
    return py_files


def generate_tests(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    prompt = f"""
            You are an expert Python developer. Generate clean PyTest-style unit tests for this code:
            Only return code, no explanation."""
    result = test_gen(prompt, max_new_tokens=512, do_sample=False)[
        0]["generated_text"]
    return result.strip()


def save_all_tests(repo_url: str):
    repo_path = clone_repo(repo_url)
    files = get_python_files(repo_path)

    test_code_blocks = []

    for f in files:
        print(f"Generating tests for {f}")
        test_code = generate_tests(f)
        test_code_blocks.append(test_code)

    # Combine and save to a single test file
    os.makedirs("output", exist_ok=True)
    with open("output/tests.py", "w", encoding="utf-8") as f:
        f.write("\n\n".join(test_code_blocks))

    print("✅ Tests saved to output/tests.py")


save_all_tests(
    "https://github.com/ausrys/ai_agents")
