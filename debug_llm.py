import importlib.util
import json
import os

from openai import OpenAI

# Load llm_summary.py directly from this folder, so Python's search path can't get in the way
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # the scripts read data/training.db relative to the project folder
path = os.path.join(HERE, "llm_summary.py")
print("Looking for:", path, "| exists:", os.path.exists(path))
print("Files with 'llm' in the name here:", [repr(f) for f in os.listdir(HERE) if "llm" in f.lower()])
spec = importlib.util.spec_from_file_location("llm_summary", path)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

weeks = L.weekly_stats(1)
if not weeks:
    raise SystemExit("No finished weeks found. Run clean_load.py first.")

client = OpenAI(base_url=L.BASE_URL, api_key="lm-studio")
resp = client.chat.completions.create(
    model=L.MODEL,
    temperature=0.2,
    max_tokens=L.MAX_TOKENS,
    messages=[
        {"role": "system", "content": L.SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(weeks[0])},
    ],
)
choice = resp.choices[0]
print("model:", L.MODEL)
print("finish reason:", choice.finish_reason)
print("content:", repr(choice.message.content))
print("reasoning:", repr(getattr(choice.message, "reasoning_content", None))[:800])
print("usage:", resp.usage)