"""Answer generation from retrieved passages with a small instruction-tuned LLM."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache

import torch

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered source passages provided by the user. "
    "Every sentence of your answer must end with the id of the passage that supports it, "
    "in square brackets, for example [12345]. Do not use outside knowledge. "
    "If the passages do not contain the information needed, reply exactly: "
    "INSUFFICIENT EVIDENCE IN SOURCES."
)
ABSTAIN_MARKER = "INSUFFICIENT EVIDENCE"
CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class Source:
    doc_id: str
    text: str
    truncated: bool


@lru_cache(maxsize=1)
def load_llm(hf_id: str, revision: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, dtype=dtype)
    model.to(device).eval()
    return tok, model


def make_sources(doc_ids: list[str], doc_text, max_words: int) -> list[Source]:
    out = []
    for d in doc_ids:
        words = doc_text(d).split()
        out.append(Source(d, " ".join(words[:max_words]), len(words) > max_words))
    return out


def build_messages(question: str, sources: list[Source]) -> list[dict]:
    block = "\n\n".join(f"[{s.doc_id}] {s.text}" for s in sources)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Source passages:\n\n{block}\n\nQuestion: {question}"},
    ]


def generate(tok, model, messages: list[dict], max_new_tokens: int, device: str) -> dict:
    """Greedy decoding (no sampling), so a rerun on the same hardware/software
    reproduces the text; different GPUs or library versions can still differ slightly."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return {"text": tok.decode(new_tokens, skip_special_tokens=True).strip(),
            "prompt_tokens": int(enc["input_ids"].shape[1]),
            "new_tokens": int(new_tokens.shape[0]),
            "seconds": round(time.perf_counter() - t0, 2)}


LEADING_CITATIONS_RE = re.compile(r"^((?:\[[^\[\]]+\]\s*)+)(.*)$", re.S)


def split_sentences(answer: str) -> list[str]:
    """Split on sentence-final punctuation, keeping citations with the sentence they follow:
    "... cancer. [123]" and "... cancer. [123] Next ..." both attach [123] to "... cancer."
    when that sentence has no citation of its own. A citation opening the answer stays put."""
    parts = [x for x in re.split(r"(?<=[.!?])\s+", answer.strip()) if x]
    out: list[str] = []
    for part in parts:
        m = LEADING_CITATIONS_RE.match(part)
        if out and m and not CITATION_RE.search(out[-1]):
            out[-1] = f"{out[-1]} {m.group(1).strip()}"
            part = m.group(2).strip()
            if not part or CITATION_RE.sub("", part).strip(" .") == "":
                continue
        out.append(part)
    return out


def citation_check(answer: str, sources: list[Source]) -> dict:
    """Mechanical checks only: which ids are cited and whether they were provided.
    Whether a cited passage *supports* the sentence must be judged by a person."""
    cited = []
    for m in CITATION_RE.findall(answer):
        cited += [c.strip() for c in m.split(",") if c.strip()]
    given = [s.doc_id for s in sources]
    sentences = split_sentences(answer)
    return {
        "abstained": ABSTAIN_MARKER in answer.upper(),
        "cited_ids": cited,
        "cited_not_provided": [c for c in cited if c not in given],
        "sentences_without_citation": [s for s in sentences
                                       if not CITATION_RE.search(s) and ABSTAIN_MARKER not in s.upper()],
    }
