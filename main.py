"""
Shannon Coder API - Optimized for Claude Code with TRUE streaming.

Uses deep model with real SSE streaming - tokens appear as generated.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from firebase_functions import https_fn

_coder_logger = logging.getLogger("shannon-coder")
_coder_logger.setLevel(logging.INFO)

CODER_MODEL_CODE = "shannon-coder-1"
BASE_MODEL_CODE = "shannon-balanced-grpo"  # Balanced model - avoids rate limits
MAX_FUNCTION_CALL_ITERATIONS = 10

# System prompt (appended to main boss prompt via system_prompt parameter)
# This gets added as "USER-PROVIDED SYSTEM PROMPT:" in the boss call
CODER_SYSTEM_PROMPT = """You are Shannon Coder, an expert AI coding assistant optimized for Claude Code CLI.

TOOL USAGE RULES (CRITICAL - FOLLOW EXACTLY):
1. When asked to explore/understand code:
   - First call Glob to find files (e.g., Glob pattern="**/*.py")
   - Then call Read on 3-5 relevant files found
   - Analyze and call more tools if needed

2. When asked to search for something:
   - Call Grep with appropriate pattern
   - Then Read the matching files

3. When asked to edit/modify code:
   - First call Read on the target file
   - Then call Edit with old_string and new_string

4. ALWAYS make multiple tool calls. One call is NOT enough.
   - Minimum 3 tool calls for exploration tasks
   - Chain: Glob → Read → Read → Read → summarize

5. NEVER just describe what you would do - ACTUALLY call the tools.

6. NEVER output tool IDs like "toolu_xxx" as plain text.

Respond by calling the appropriate tools. Be direct. Complete the task fully."""

# Kept for backwards compatibility but not used - we append via system_prompt instead
CODER_BOSS_PROMPT_TEMPLATE = None


def _core():
    import main as core
    return core


def _api_endpoints():
    import api_endpoints
    return api_endpoints


def _stream_response_direct(
    client: Any,
    prompt: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    length: int = 16384,
) -> Generator[str, None, None]:
    """
    Stream directly from Shannon AI - no refuser, minimal latency.
    Yields text chunks as they arrive.
    """
    core = _core()
    from google.genai import types

    # Build prompt with system instructions inline
    full_prompt = f"{system_prompt}\n\n---\n\nUSER REQUEST:\n{prompt}"

    # Build history
    history_contents = core._build_history_contents(history) if history else []

    # Build contents
    user_parts = [types.Part.from_text(text=full_prompt)]
    contents = history_contents + [types.Content(role="user", parts=user_parts)]

    # Config - no thinking, fast response
    cfg = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=length,
        safety_settings=core._SAFETY_OFF,
    )

    # Get model
    cfg_data = core._resolve_boss_config(BASE_MODEL_CODE)
    model_name = cfg_data["model"]

    _coder_logger.info("Streaming from model: %s", model_name)

    try:
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=cfg,
        ):
            texts = core._extract_texts_from_response(chunk)
            for text in texts:
                if text:
                    yield text
    except Exception as exc:
        _coder_logger.exception("Stream error: %s", exc)
        yield f"\n\n[Error: {exc}]"


def _generate_anthropic_stream(
    uid: str,
    prompt: str,
    history: List[Dict[str, str]],
    system_prompt: Optional[str],
    response_id: str,
    model_name: str,
) -> Generator[bytes, None, None]:
    """Generate Anthropic SSE stream - optimized for Claude Code."""
    core = _core()

    full_system = CODER_SYSTEM_PROMPT
    if system_prompt:
        full_system = f"{CODER_SYSTEM_PROMPT}\n\n{system_prompt}"

    try:
        client = core._get_genai_client()

        # message_start
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': response_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n".encode()

        # content_block_start
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode()

        # Stream content - each chunk immediately
        for text_chunk in _stream_response_direct(client, prompt, full_system, history):
            if text_chunk:
                delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text_chunk}}
                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode()

        # content_block_stop
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()

        # message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 0}})}\n\n".encode()

        # message_stop
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode()

    except Exception as exc:
        _coder_logger.exception("Anthropic stream error: %s", exc)
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'server_error', 'message': str(exc)}})}\n\n".encode()


def _generate_openai_stream(
    uid: str,
    prompt: str,
    history: List[Dict[str, str]],
    system_prompt: Optional[str],
    response_id: str,
    created_ts: int,
    model_name: str,
) -> Generator[bytes, None, None]:
    """Generate OpenAI SSE stream."""
    core = _core()

    full_system = CODER_SYSTEM_PROMPT
    if system_prompt:
        full_system = f"{CODER_SYSTEM_PROMPT}\n\n{system_prompt}"

    try:
        client = core._get_genai_client()

        # Stream content
        for text_chunk in _stream_response_direct(client, prompt, full_system, history):
            if text_chunk:
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n".encode()

        # Done
        done_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    except Exception as exc:
        _coder_logger.exception("OpenAI stream error: %s", exc)
        yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n".encode()


def run_coder_chat(
    uid: str,
    prompt: str,
    history: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    function_definitions: Optional[List[Dict[str, Any]]] = None,
    function_call_control: Optional[Union[str, Dict[str, str]]] = None,
    length: int = 16384,
) -> Tuple[str, Dict[str, int], Optional[Dict[str, Any]], int]:
    """Non-streaming coder chat with tool support."""
    api = _api_endpoints()

    full_system = CODER_SYSTEM_PROMPT
    if system_prompt:
        full_system = f"{CODER_SYSTEM_PROMPT}\n\n{system_prompt}"

    if function_definitions is None:
        function_definitions = []
    if api._should_include_web_search_function(uid, function_definitions):
        function_definitions.append(api.WEB_SEARCH_FUNCTION)

    return api._run_ego_chat_with_tools(
        uid, prompt, history,
        model_code=BASE_MODEL_CODE,
        length=length,
        origin=None,
        system_prompt=full_system,
        function_definitions=function_definitions if function_definitions else None,
        function_call_control=function_call_control,
    )


def handle_anthropic_coder_request(
    req: https_fn.Request,
    uid: str,
    body: Dict[str, Any],
    origin: Optional[str],
    allow_headers: Optional[str],
) -> https_fn.Response:
    """Handle Anthropic request - optimized for Claude Code."""
    api = _api_endpoints()
    core = _core()

    stream_requested = body.get("stream", False)

    # Extract prompt
    prompt_value: Optional[str] = None
    history: List[Dict[str, str]] = []
    system_prompt = str(body.get("system") or "").strip() or None

    try:
        if body.get("prompt") is not None:
            prompt_value = str(body.get("prompt") or "")
        else:
            msg_system, normalized_messages = api._extract_system_prompt(body.get("messages"))
            prompt_value, history = api._messages_to_prompt_history(normalized_messages)
            if msg_system:
                system_prompt = f"{system_prompt}\n{msg_system}" if system_prompt else msg_system
    except ValueError as exc:
        return api._build_anthropic_error(str(exc), 400, origin, allow_headers)

    if not prompt_value or not prompt_value.strip():
        return api._build_anthropic_error("Prompt required", 400, origin, allow_headers)

    length = api._coerce_length(body.get("max_tokens") or body.get("max_tokens_to_sample"))

    try:
        core._apply_quota(uid, length)
    except PermissionError:
        return api._build_anthropic_error("Quota exceeded", 402, origin, allow_headers)

    response_id = f"msg_{secrets.token_hex(12)}"
    model_name = body.get("model") or CODER_MODEL_CODE

    # Check for tools - if present, must use non-streaming to support tool calls
    raw_tools = body.get("tools")
    tools = api._normalize_anthropic_tools(raw_tools) if raw_tools else []
    has_tools = bool(tools)

    # STREAMING - only if no tools (tools require non-streaming path)
    if stream_requested and not has_tools:
        _coder_logger.info("Starting Anthropic stream for Claude Code")

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        if origin:
            headers["Access-Control-Allow-Origin"] = origin
        else:
            headers["Access-Control-Allow-Origin"] = "*"
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        headers["Access-Control-Allow-Headers"] = allow_headers or "*"

        return https_fn.Response(
            response=_generate_anthropic_stream(uid, prompt_value, history, system_prompt, response_id, model_name),
            status=200,
            headers=headers,
        )

    # NON-STREAMING with tools (also used when streaming + tools requested)
    if has_tools:
        _coder_logger.info("Using non-streaming path for tool support (%d tools)", len(tools))

    tool_choice = body.get("tool_choice")
    function_call_control: Optional[Union[str, Dict[str, str]]] = "auto"
    if tool_choice:
        if isinstance(tool_choice, str):
            function_call_control = "none" if tool_choice == "none" else "auto"
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") == "tool" and tool_choice.get("name"):
                function_call_control = {"name": tool_choice["name"]}

    current_prompt = prompt_value
    current_history = list(history)
    current_tools = list(tools) if tools else []
    accumulated_usage = {"input_tokens": 0, "output_tokens": 0}

    for _ in range(MAX_FUNCTION_CALL_ITERATIONS):
        response_text, usage, function_call, status_code = run_coder_chat(
            uid, current_prompt, current_history,
            system_prompt=system_prompt,
            function_definitions=current_tools if current_tools else None,
            function_call_control=function_call_control,
            length=length,
        )

        if status_code >= 400:
            return api._build_anthropic_error("Chat failed", status_code, origin, allow_headers)

        accumulated_usage["input_tokens"] += usage.get("prompt_tokens", 0)
        accumulated_usage["output_tokens"] += usage.get("response_tokens", 0)

        if function_call:
            func_name = function_call["name"]
            func_args = function_call["arguments"]

            if func_name == "web_search":
                func_result, success = api._execute_builtin_function(func_name, func_args)
                if success:
                    try:
                        core._apply_search_quota(uid, 1)
                    except PermissionError:
                        func_result = json.dumps({"error": "Search quota exceeded"})

                current_history.append({"role": "assistant", "content": json.dumps({"function_call": function_call})})
                current_history.append({"role": "user", "content": f"[Tool Result: {func_result}]"})
                current_prompt = f"{prompt_value}\n\n[{func_name}: {func_result}]"
                continue

            # External tool
            try:
                args_dict = json.loads(func_args) if func_args else {}
            except:
                args_dict = {}

            return api._cors_response({
                "id": response_id, "type": "message", "role": "assistant", "model": model_name,
                "content": [{"type": "tool_use", "id": f"toolu_{secrets.token_hex(12)}", "name": func_name, "input": args_dict}],
                "stop_reason": "tool_use", "usage": accumulated_usage,
            }, origin=origin, allow_headers=allow_headers)

        return api._cors_response({
            "id": response_id, "type": "message", "role": "assistant", "model": model_name,
            "content": [{"type": "text", "text": response_text}],
            "stop_reason": "end_turn", "usage": accumulated_usage,
        }, origin=origin, allow_headers=allow_headers)

    return api._build_anthropic_error("Max iterations", 500, origin, allow_headers)


def handle_openai_coder_request(
    req: https_fn.Request,
    uid: str,
    body: Dict[str, Any],
    origin: Optional[str],
    allow_headers: Optional[str],
) -> https_fn.Response:
    """Handle OpenAI request."""
    api = _api_endpoints()
    core = _core()

    stream_requested = body.get("stream", False)

    try:
        system_prompt, normalized_messages = api._extract_system_prompt(body.get("messages"))
        prompt, history = api._messages_to_prompt_history(normalized_messages)
    except ValueError as exc:
        return api._build_openai_error(str(exc), 400, origin, allow_headers)

    length = api._coerce_length(body.get("max_tokens") or body.get("max_tokens_to_sample"))

    try:
        core._apply_quota(uid, length)
    except PermissionError:
        return api._build_openai_error("Quota exceeded", 402, origin, allow_headers)

    response_id = f"chatcmpl-{secrets.token_hex(12)}"
    created_ts = int(datetime.now(timezone.utc).timestamp())
    model_name = body.get("model") or CODER_MODEL_CODE

    # STREAMING
    if stream_requested:
        _coder_logger.info("Starting OpenAI stream")

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        if origin:
            headers["Access-Control-Allow-Origin"] = origin
        else:
            headers["Access-Control-Allow-Origin"] = "*"
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        headers["Access-Control-Allow-Headers"] = allow_headers or "*"

        return https_fn.Response(
            response=_generate_openai_stream(uid, prompt, history, system_prompt, response_id, created_ts, model_name),
            status=200,
            headers=headers,
        )

    # NON-STREAMING with tools
    raw_functions = body.get("functions")
    raw_tools = body.get("tools")
    use_tools_format = raw_tools is not None

    functions = api._normalize_functions(raw_functions, raw_tools)
    function_call_control = api._normalize_function_call_control(body.get("function_call"), body.get("tool_choice"))

    current_prompt = prompt
    current_history = list(history)
    current_functions = list(functions) if functions else []
    accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for _ in range(MAX_FUNCTION_CALL_ITERATIONS):
        response_text, usage, function_call, status_code = run_coder_chat(
            uid, current_prompt, current_history,
            system_prompt=system_prompt,
            function_definitions=current_functions if current_functions else None,
            function_call_control=function_call_control,
            length=length,
        )

        if status_code >= 400:
            return api._build_openai_error("Chat failed", status_code, origin, allow_headers)

        accumulated_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        accumulated_usage["completion_tokens"] += usage.get("response_tokens", 0)
        accumulated_usage["total_tokens"] += usage.get("total_tokens", 0)

        if function_call:
            func_name = function_call["name"]
            func_args = function_call["arguments"]

            if func_name == "web_search":
                func_result, success = api._execute_builtin_function(func_name, func_args)
                if success:
                    try:
                        core._apply_search_quota(uid, 1)
                    except PermissionError:
                        func_result = json.dumps({"error": "Search quota exceeded"})

                current_history.append({"role": "assistant", "content": json.dumps({"function_call": function_call})})
                current_history.append({"role": "function", "name": func_name, "content": func_result})
                current_prompt = f"{prompt}\n\n[{func_name}: {func_result}]"
                continue

            # External tool
            if use_tools_format:
                return api._cors_response({
                    "id": response_id, "object": "chat.completion", "created": created_ts, "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                        "tool_calls": [{"id": f"call_{secrets.token_hex(12)}", "type": "function",
                            "function": {"name": func_name, "arguments": func_args}}]},
                        "finish_reason": "tool_calls"}],
                    "usage": accumulated_usage,
                }, origin=origin, allow_headers=allow_headers)
            else:
                return api._cors_response({
                    "id": response_id, "object": "chat.completion", "created": created_ts, "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                        "function_call": {"name": func_name, "arguments": func_args}},
                        "finish_reason": "function_call"}],
                    "usage": accumulated_usage,
                }, origin=origin, allow_headers=allow_headers)

        return api._cors_response({
            "id": response_id, "object": "chat.completion", "created": created_ts, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
            "usage": accumulated_usage,
        }, origin=origin, allow_headers=allow_headers)

    return api._build_openai_error("Max iterations", 500, origin, allow_headers)


def is_coder_model(model: Optional[str]) -> bool:
    if not model:
        return False
    m = model.strip().lower()
    return m in ("shannon-coder-1", "shannon-coder", "coder-1", "coder")
