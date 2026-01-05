from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict, Any, Optional
import json

import os
from openai import OpenAI

from src.chroma_api_bge_local import ChromaDB, QueryHit

class OpenAICompatibleChat:
    def __init__(
        self,
        *,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url_env: str = "LLM_BASE_URL",
        model_env: str = "LLM_MODEL",
        default_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model: str = "qwen-plus",
        temperature: float = 0.2,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key: please set {api_key_env}.")

        base_url = os.getenv(base_url_env, default_base_url)
        model = os.getenv(model_env, default_model)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def __call__(self, messages: List[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

@dataclass
class RAGConfig:
    top_k: int = 5
    max_context_chars: int = 4000

@dataclass
class RAGResult:
    question: str
    answer: str
    hits: List[QueryHit]
    context: str
    messages: List[dict]


def build_context(hits: List[QueryHit], max_chars: int = 4000) -> Tuple[str, List[QueryHit]]:
    blocks: List[str] = []
    used: List[QueryHit] = []
    total = 0

    for i, h in enumerate(hits, start=1):
        block = (
            f"[{i}] source={h.metadata.get('source')} chunk_id={h.id} "
            f"doc_id={h.metadata.get('doc_id')} chunk={h.metadata.get('chunk_index')} "
            f"distance={h.distance:.4f}\n"
            f"{h.document}\n"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        used.append(h)
        total += len(block)

    return "\n".join(blocks).strip(), used


def make_rag_messages(question: str, context: str, prompt: str) -> List[dict]:
    user = (
        f"【用户问题】\n{question}\n\n"
        f"【上下文】\n{context if context else '（无召回内容）'}\n\n"
        "请给出回答："
    )
    return [{"role": "system", "content": prompt}, {"role": "user", "content": user}]


def rag_ask(
    *,
    db: ChromaDB,
    llm_chat: Callable[[List[dict]], str],
    prompt: str,
    question: str,
    top_k: int = 5,
    max_context_chars: int = 4000,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hits = db.query(question, top_k=top_k, where=where)
    context, used_hits = build_context(hits, max_chars=max_context_chars)
    messages = make_rag_messages(question, context, prompt)
    answer = llm_chat(messages)

    return {
        "question": question,
        "answer": answer,
        "hits": used_hits,
        "context": context,
        "messages": messages,
    }

def hit_to_dict(h: QueryHit) -> dict:
    return {
        "id": h.id,
        "distance": h.distance,
        "metadata": h.metadata,
        "document_preview": (h.document[:200] + "...") if h.document and len(h.document) > 200 else h.document,
    }

class RAGEngine:
    def __init__(
        self,
        *,
        db: ChromaDB,
        llm_chat: Callable[[List[dict]], str],
        prompt: str,
        config: Optional[RAGConfig] = None,
    ):
        self.db = db
        self.llm_chat = llm_chat
        self.prompt = prompt
        self.config = config or RAGConfig()

    def ask(
            self,
            question: str,
            *,
            prompt: Optional[str] = None,
            top_k: Optional[int] = None,
            max_context_chars: Optional[int] = None,
            where: Optional[Dict[str, Any]] = None,
            max_distance: Optional[float] = None,
            min_keep: int = 1,
    ) -> RAGResult:
        _prompt = prompt if prompt is not None else self.prompt
        _top_k = top_k if top_k is not None else self.config.top_k
        _max_chars = max_context_chars if max_context_chars is not None else self.config.max_context_chars

        hits = self.db.query(question, top_k=_top_k, where=where)

        if max_distance is not None:
            filtered = [h for h in hits if getattr(h, "distance", None) is not None and h.distance <= max_distance]
            if len(filtered) >= min_keep:
                hits = filtered
            else:
                hits = hits[:min_keep]  # 保底

        context, used_hits = build_context(hits, max_chars=_max_chars)

        print("=== USED HITS ===")
        print(json.dumps([hit_to_dict(h) for h in used_hits], ensure_ascii=False, indent=2))

        messages = make_rag_messages(question, context, _prompt)
        answer = self.llm_chat(messages)

        return RAGResult(
            question=question,
            answer=answer,
            hits=used_hits,
            context=context,
            messages=messages,
        )

    def ask_without_search(self, question: str, *, prompt: Optional[str] = None) -> RAGResult:
        _prompt = prompt if prompt is not None else self.prompt
        context = ""
        messages = make_rag_messages(question, context, _prompt)
        answer = self.llm_chat(messages)

        return RAGResult(
            question=question,
            answer=answer,
            hits="",
            context="",
            messages=messages,
        )