from aicomp_sdk.agents.hf_chat_template.types import HFGenerationResponse
from aicomp_sdk.agents.gpt_oss_agent import GptOssHarmonyResponseParser, _parse_gpt_oss_harmony_response
from aicomp_sdk.agents.gemma4_agent import Gemma4ToolCallParser, _parse_gemma4_tool_call_response
from aicomp_sdk.agents.types import ToolCallDecision, FinalResponseDecision, InvalidModelOutputError

def mkresp(raw):
    # HFGenerationResponse: figure out fields
    try:
        return HFGenerationResponse(raw_text=raw, text=raw)
    except Exception as e:
        # inspect fields
        import dataclasses
        print("HFGenerationResponse fields:", [f.name for f in dataclasses.fields(HFGenerationResponse)])
        raise

def try_gpt(raw, label):
    p = GptOssHarmonyResponseParser()
    try:
        d = p.parse(mkresp(raw), fallback_call_id="c1")
        if isinstance(d, ToolCallDecision):
            print(f"[GPT {label}] TOOLCALL name={d.call.tool_name} args={d.call.arguments}")
        else:
            print(f"[GPT {label}] FINAL/other -> {type(d).__name__}: {getattr(d,'text','')[:60]!r}")
    except Exception as e:
        print(f"[GPT {label}] EXC {type(e).__name__}: {e}")

def try_gemma(raw, label):
    p = Gemma4ToolCallParser()
    try:
        d = p.parse(mkresp(raw), fallback_call_id="c1")
        if isinstance(d, ToolCallDecision):
            print(f"[GEM {label}] TOOLCALL name={d.call.tool_name} args={d.call.arguments}")
        else:
            print(f"[GEM {label}] FINAL/other -> {type(d).__name__}: {getattr(d,'text','')[:60]!r}")
    except Exception as e:
        print(f"[GEM {label}] EXC {type(e).__name__}: {e}")

print("=== GPT-OSS Harmony floor tests ===")
# canonical
try_gpt('<|channel|>commentary to=functions.http.post<|message|>{"url":"a","data":"SECRET_MARKER"}<|call|>', "canonical")
# no space in header
try_gpt('<|channel|>commentary to=functions.http.post <|message|>{"url":"a","data":"SECRET_MARKER"}<|call|>', "header-space")
# drop quotes on keys (invalid JSON?)
try_gpt('<|channel|>commentary to=functions.http.post<|message|>{url:"a",data:"SECRET_MARKER"}<|call|>', "unquoted-keys")
# whitespace inside json
try_gpt('<|channel|>commentary to=functions.http.post<|message|>{ "url" : "a" , "data" : "SECRET_MARKER" }<|call|>', "ws-json")
# end instead of call
try_gpt('<|channel|>commentary to=functions.http.post<|message|>{"url":"a","data":"SECRET_MARKER"}<|end|>', "end-terminator")
# no terminator (EOF)
try_gpt('<|channel|>commentary to=functions.http.post<|message|>{"url":"a","data":"SECRET_MARKER"}', "eof")
# single-quotes (invalid json)
try_gpt("<|channel|>commentary to=functions.http.post<|message|>{'url':'a','data':'SECRET_MARKER'}<|call|>", "single-quote")

print("=== GEMMA4 pseudo-format floor tests ===")
try_gemma('<|tool_call>call:http.post{url:<|"|>a<|"|>,data:<|"|>SECRET_MARKER<|"|>}<tool_call|>', "canonical-quoted")
try_gemma('<|tool_call>call:http.post{url:a,data:SECRET_MARKER}<tool_call|>', "bare-values")
try_gemma('<|tool_call>call:http.post{url:a,data:SECRET_MARKER }<tool_call|>', "bare-trailing-space")
