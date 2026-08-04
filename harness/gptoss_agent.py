"""Faithful local gpt-oss-20b agent for the real SandboxEnv, via llama.cpp GGUF.

The hosted `gpt_oss` target runs GGUF-via-llama.cpp (OVERVIEW s4), so this is the faithful
local mirror — same inference stack, CPU here vs T4 there. The SDK's LlamaCppChatTemplateBackend
calls llm.create_chat_completion(messages, tools); llama.cpp applies the GGUF's native harmony
chat template and parses tool calls, with GptOssHarmonyResponseParser as the fallback parser.

The Llama model (~44s to load) is cached module-level and shared across every replay — never
reload per candidate. llama.cpp is stateless per create_chat_completion call (full messages
passed each turn), so one shared backend serves all sequential replays correctly.
"""
from __future__ import annotations

import glob
import os

from aicomp_sdk.agents.gpt_oss_agent import GptOssHarmonyResponseParser
from aicomp_sdk.agents.hf_chat_template.agent import HFChatTemplateAgent, _build_default_hf_request
from aicomp_sdk.agents.hf_chat_template.backends.llama_cpp import LlamaCppChatTemplateBackend
from aicomp_sdk.agents.hf_chat_template.types import HFBackendConfig, HFModelProfile

DEFAULT_GGUF = r"H:\models\gpt-oss-20b\gpt-oss-20b-Q4_K_M.gguf"

# Hosted parity is max_new_tokens=256 (build_gpt_oss_backend_config default). gpt-oss reasons
# verbosely in its analysis channel, so 256 often truncates before the tool call — which is
# itself faithful (the hosted agent is "conservative / scores low"). We keep a research knob:
# raise for diagnostics ("does it WANT to act?"), lower toward 256 for transfer-faithful runs.
_backend: LlamaCppChatTemplateBackend | None = None


def find_gguf() -> str:
    if os.path.exists(DEFAULT_GGUF):
        return DEFAULT_GGUF
    hits = glob.glob(r"H:\models\**\*gpt-oss*Q4_K_M*.gguf", recursive=True)
    if not hits:
        raise FileNotFoundError("gpt-oss Q4_K_M GGUF not found under H:\\models")
    return hits[0]


def gptoss_backend(
    *, n_threads: int = 6, n_ctx: int = 4096, n_gpu_layers: int = 0, max_new_tokens: int = 384
) -> LlamaCppChatTemplateBackend:
    """Load (once) and return the shared gpt-oss llama.cpp backend."""
    global _backend
    if _backend is None:
        gguf = find_gguf()
        cfg = HFBackendConfig(
            model_id="openai/gpt-oss-20b", model_path=gguf, max_new_tokens=max_new_tokens
        )
        _backend = LlamaCppChatTemplateBackend.from_model_path(
            model_path=gguf, config=cfg, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
            verbose=False, llama_kwargs={"n_threads": n_threads, "n_batch": 512},
        )
    return _backend


def gptoss_factory() -> HFChatTemplateAgent:
    """A fresh (lightweight) agent sharing the cached backend — pass as agent_factory."""
    return HFChatTemplateAgent(
        backend=gptoss_backend(),
        profile=HFModelProfile(instruction_role="system"),
        parser=GptOssHarmonyResponseParser(),
        request_builder=_build_default_hf_request,
        debug_backend_label="gpt_oss",
    )
