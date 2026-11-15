"""A tiny OpenAI-compatible /v1/chat/completions server in front of the JAX model (Module 39).

Tool calling over the OpenAI API is a *server-side convention*: the server writes the tool list into the prompt,
decodes the model's reply under constraints, and returns structured `tool_calls`. LM Studio, llama.cpp and vLLM
all do a version of this. Any OpenAI client — LangGraph, the openai SDK, curl — can then drive the JAX model.

    srv = start(lm, ex, port=8765)      # runs uvicorn in a background thread
    ... ChatOpenAI(base_url="http://127.0.0.1:8765/v1", api_key="none", model="smollm2-360m-jax") ...
    srv.should_exit = True
"""
from __future__ import annotations

import json
import threading
import time
import uuid

KINDS = {"date": "date", "start": "date", "end": "date", "expression": "expression"}


def _tools_from_openai(specs, A) -> dict:
    tools = {}
    for spec in specs or []:
        f = spec["function"]
        props = f.get("parameters", {}).get("properties", {})
        params = [(k, "integer" if v.get("type") in ("integer", "number") else KINDS.get(k, "text"))
                  for k, v in props.items()]
        tools[f["name"]] = A.Tool(f["name"], f.get("description", "").strip().split("\n")[0], params, None)
    return tools


def _render(messages, tools, A) -> str:
    out = ""
    if tools:
        sys_extra = next((m["content"] for m in messages if m["role"] == "system"), None)
        out = A.system_prompt(tools, A.EXAMPLES + A.DATE_EXAMPLES + A.LIBRARY_EXAMPLES,
                              A.SYSTEM + ("\n" + sys_extra if sys_extra else ""))
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            if not tools:
                out += A.turn("system", content)
        elif role == "assistant" and m.get("tool_calls"):
            calls = "".join(f'<tool_call>{{"name": "{c["function"]["name"]}", "arguments": '
                            f'{c["function"]["arguments"]}}}</tool_call>' for c in m["tool_calls"])
            out += A.turn("assistant", calls)
        elif role == "tool":
            out += A.turn("user", f"<tool_result>{content}</tool_result>")
        elif role in ("user", "assistant"):
            out += A.turn(role, content)
    return out + A.ASSISTANT


def make_app(lm, ex, A, model_name: str = "smollm2-360m-jax"):
    from fastapi import FastAPI

    app = FastAPI()
    lock = threading.Lock()                     # one GPU, one request at a time

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "learn-jax"}]}

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        tools = _tools_from_openai(body.get("tools"), A)
        with lock:
            state = lm.feed(lm.new(), _render(body["messages"], tools, A))
            if tools:   # calibrate the tool choice against a content-free question (lesson section 3)
                neutral = [m for m in body["messages"] if m["role"] == "system"] + [{"role": "user", "content": "N/A"}]
                base = lm.feed(lm.new(), _render(neutral, tools, A))
                text, _ = A.constrained_turn(lm, ex, state, tools, base=base)
            else:
                text, _, _ = lm.generate(state, int(body.get("max_tokens") or 200))
        call = ex.parse_tool_call(text) if tools else None
        if call:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}
            finish = "tool_calls"
        else:
            m = A.ANSWER.search(text)
            msg, finish = {"role": "assistant", "content": (m.group(1) if m else text).strip()}, "stop"
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
                "model": model_name, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": state.pos, "completion_tokens": lm.count(text),
                          "total_tokens": state.pos + lm.count(text)}}

    return app


def start(lm, ex, A, port: int = 8765):
    """Serve in a background thread; returns the uvicorn.Server (set .should_exit = True to stop)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(make_app(lm, ex, A), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    return server
