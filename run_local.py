import os
from openai import OpenAI

_omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1"),
               api_key=os.getenv("OMLX_API_KEY", "not-used"))
_vmlx = OpenAI(base_url=os.getenv("VMLX_BASE_URL", "http://127.0.0.1:8003/v1"),
               api_key=os.getenv("VMLX_API_KEY", "not-used"))

TIERS = {
    "opus": (_omlx, os.getenv("MODEL_OPUS", "Qwen3.6-35B-A3B-nvfp4")),
    "sonnet": (_omlx, os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")),
    "haiku": (_omlx, os.getenv("MODEL_HAIKU", "gpt-oss-20b-MXFP4-Q8")),
    "vmlx": (_vmlx, os.getenv("MODEL_VMLX", "gemma-4-31B-uncensored-heretic-mlx-4bit")),
}

def chat(tier: str, messages: list[dict], **kwargs) -> str:
    client, model = TIERS[tier]
    r = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return r.choices[0].message.content