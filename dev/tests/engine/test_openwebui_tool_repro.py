"""Repro suite for issue #214 (OpenWebUI + `splash serve` tool calling).

OpenWebUI defaults to native function calling: it injects its builtin
``search_web``/``fetch_url`` tools as standard OpenAI ``tools`` entries in the
/v1/chat/completions body and consumes streamed ``delta.tool_calls`` chunks.
These tests replay that exact client contract against the fake-runtime harness:

* the tools survive normalization and reach the chat template prompt,
* the streamed call carries the OpenAI delta shape OpenWebUI parses
  (``index``/``id``/``type``/``function.name``, then argument fragments),
* buffered (non-string) arguments keep the stream alive with data-frame
  keepalives instead of SSE comments (b97e439),
* the call finishes with ``finish_reason: "tool_calls"``,
* OpenWebUI's extra top-level body fields are tolerated, not rejected.
"""

import json
import unittest
from unittest import mock

from jsonschema import Draft202012Validator

from dev.tests.test_server import (
    FakeRuntime,
    FakeTokenizer,
    Harness,
    Plan,
)
from server import server as api

# Tool specs shaped exactly like OpenWebUI's builtin tools (tools/builtin.py),
# whose JSON specs are generated from these signatures/docstrings.
SEARCH_WEB = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the public web for information. Best for current events, "
            "external references, or topics not covered in internal documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up",
                },
                "count": {
                    "type": "integer",
                    "description": (
                        "Number of results to return (default: "
                        "admin-configured value)"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}
FETCH_URL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch the contents of a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
            },
            "required": ["url"],
        },
    },
}
OWUI_TOOLS = [SEARCH_WEB, FETCH_URL]

QUERY = "latest splash engine releases"
# What the Qwen tool template renders a search_web call into on the wire.
# Built by concatenation so this source file never contains a literal
# tool-call frame.
LT = "<"
CALL_QUERY_ONLY = (
    f"{LT}tool_call>\n{LT}function=search_web>\n{LT}parameter=query>\n"
    f"{QUERY}\n{LT}/parameter>\n{LT}/function>\n{LT}/tool_call>\n"
)
CALL_WITH_COUNT = (
    f"{LT}tool_call>\n{LT}function=search_web>\n{LT}parameter=query>\n"
    f"{QUERY}\n{LT}/parameter>\n{LT}parameter=count>\n3\n{LT}/parameter>\n"
    f"{LT}/function>\n{LT}/tool_call>\n"
)


class OpenWebUIToolReproTests(unittest.TestCase):
    """Issue #214: OpenWebUI builtin tools over `splash serve`."""

    def setUp(self):
        self.harnesses = []

    def tearDown(self):
        for harness in reversed(self.harnesses):
            harness.close()

    def harness(self, plan, fragments):
        tokenizer = FakeTokenizer()
        for token, text in fragments.items():
            tokenizer.fragments[token] = text
        runtime = FakeRuntime(plan)
        harness = Harness(runtime, tokenizer=tokenizer, max_context=4096)
        self.harnesses.append(harness)
        return harness

    @staticmethod
    def owui_body(**extra):
        """A /v1/chat/completions body shaped like OpenWebUI's native path."""
        body = {
            "model": "test-model",
            "messages": [
                {
                    "role": "system",
                    "content": "You are Open WebUI, a helpful AI assistant.",
                },
                {"role": "user", "content": "What is new in splash releases?"},
            ],
            "stream": False,
            # OpenWebUI exposes this as the model's "Reasoning" toggle; the
            # fake tokenizer only closes its think frame when it is off.
            "reasoning_effort": "none",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "tools": OWUI_TOOLS,
            # OpenWebUI keeps non-OpenAI extras in form_data; they must not
            # turn the request into a 400.
            "metadata": {"chat_id": "c-214", "message_id": "m-1"},
        }
        body.update(extra)
        return body

    @staticmethod
    def data_chunks(payload):
        return [
            json.loads(line[len("data: ") :])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]

    def test_openwebui_tools_reach_the_model_prompt(self):
        harness = self.harness(Plan([[40]]), {40: CALL_QUERY_ONLY})
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.owui_body()
        )
        self.assertEqual(status, 200, payload)
        # The template must actually be handed both builtin tools, or the
        # model can never see search_web ("can't access the tool").
        _, kwargs = harness.tokenizer.templates[-1]
        self.assertEqual(
            [tool["function"]["name"] for tool in kwargs["tools"]],
            ["search_web", "fetch_url"],
        )
        message = json.loads(payload)["choices"][0]["message"]
        call = message["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertTrue(call["id"].startswith("call_"))
        arguments = json.loads(call["function"]["arguments"])
        self.assertEqual(arguments, {"query": QUERY})
        Draft202012Validator(
            SEARCH_WEB["function"]["parameters"]
        ).validate(arguments)

    def test_streaming_search_web_call_follows_the_openai_delta_contract(self):
        harness = self.harness(Plan([[40]]), {40: CALL_QUERY_ONLY})
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.owui_body(stream=True),
        )
        self.assertEqual(status, 200, payload)
        text = payload.decode()
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))
        chunks = self.data_chunks(payload)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        tool_deltas = [
            delta
            for chunk in chunks
            for delta in chunk["choices"][0]["delta"].get("tool_calls", [])
        ]
        self.assertTrue(tool_deltas, "no tool_calls deltas were streamed")
        header = tool_deltas[0]
        self.assertEqual(header["index"], 0)
        self.assertTrue(header["id"].startswith("call_"))
        self.assertEqual(header["type"], "function")
        self.assertEqual(header["function"], {"name": "search_web"})
        arguments = "".join(
            delta["function"].get("arguments", "") for delta in tool_deltas[1:]
        )
        self.assertEqual(json.loads(arguments), {"query": QUERY})
        finish = [
            chunk["choices"][0]["finish_reason"]
            for chunk in chunks
            if chunk["choices"][0].get("finish_reason") is not None
        ]
        self.assertEqual(finish, ["tool_calls"])

    def test_keepalive_is_a_data_frame_while_arguments_buffer(self):
        # A non-string parameter (count) is buffered until the call closes,
        # so the stream goes quiet mid-call. OpenWebUI-side stall timers only
        # see data frames; b97e439 made the keepalive an empty-delta chunk.
        plan = Plan([[41]], block=True)
        harness = self.harness(plan, {41: CALL_WITH_COUNT})
        with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
            connection, response = harness.open_stream(
                "/v1/chat/completions",
                self.owui_body(stream=True),
            )
            try:
                role = json.loads(self.next_sse_data(response))
                self.assertEqual(role["choices"][0]["delta"]["role"], "assistant")
                # The next keepalive must be a regular data chunk, not an
                # SSE comment line a data-event-only reader would skip.
                heartbeat = json.loads(self.next_sse_data(response))
                self.assertEqual(heartbeat["choices"][0]["delta"], {})
                self.assertIsNone(heartbeat["choices"][0]["finish_reason"])
            finally:
                plan.release.set()
                payload = response.read()
                connection.close()
        chunks = self.data_chunks(payload)
        arguments = "".join(
            delta["function"].get("arguments", "")
            for chunk in chunks
            for delta in chunk["choices"][0]["delta"].get("tool_calls", [])
        )
        self.assertEqual(json.loads(arguments), {"query": QUERY, "count": 3})
        finish = [
            chunk["choices"][0]["finish_reason"]
            for chunk in chunks
            if chunk["choices"][0].get("finish_reason") is not None
        ]
        self.assertEqual(finish, ["tool_calls"])

    @staticmethod
    def next_sse_data(response):
        data = []
        while True:
            raw = response.readline()
            if not raw:
                return None
            line = raw.decode().rstrip("\r\n")
            if not line:
                if data:
                    return "\n".join(data)
                continue
            if line.startswith("data:"):
                data.append(line[5:].lstrip())

    def test_second_turn_with_tool_result_history_completes(self):
        # OpenWebUI re-sends the conversation with the assistant tool_calls
        # message and the tool result attached; that history shape must be
        # accepted and the follow-up answer streamed as plain content.
        history = [
            {
                "role": "system",
                "content": "You are Open WebUI, a helpful AI assistant.",
            },
            {"role": "user", "content": "What is new in splash releases?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_214",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": QUERY}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_214",
                "content": json.dumps(
                    [{"title": "Splash 1.1.0", "link": "u", "snippet": "s"}]
                ),
            },
            {"role": "user", "content": "Summarize the results."},
        ]
        harness = self.harness(Plan([[4]]), {})
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.owui_body(messages=history, tools=OWUI_TOOLS),
        )
        self.assertEqual(status, 200, payload)
        message = json.loads(payload)["choices"][0]["message"]
        self.assertNotIn("tool_calls", message)
        self.assertEqual(message["content"].strip(), "plain answer")


if __name__ == "__main__":
    unittest.main()
