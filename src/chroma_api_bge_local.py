import os
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


class BGEEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,   # "cpu" / "cuda" / "mps"
        normalize: bool = True,
        batch_size: int = 64,
    ):
        if not model_path:
            raise ValueError("./hf_models/bge-base-zh-v1.5")

        self.model_path = model_path
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size

        # SentenceTransformer 支持本地路径
        self.model = SentenceTransformer(model_path)
        if device:
            self.model = self.model.to(device)

    def __call__(self, input: List[str]) -> List[List[float]]:
        emb = self.model.encode(
            input,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return emb.tolist()


@dataclass
class AddResult:
    doc_id: str
    chunk_ids: List[str]
    chunks_added: int


@dataclass
class QueryHit:
    id: str
    distance: float
    document: str
    metadata: Dict[str, Any]


class ChromaDB:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_function: EmbeddingFunction,
        space: str = "cosine",
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": space},
        )
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def stats(self) -> Dict[str, Any]:
        return {
            "persist_dir": self.persist_dir,
            "collection_name": self.collection.name,
            "count": self.collection.count(),
        }

    def add_text(
            self,
            text: str,
            *,
            source: str = "manual",
            doc_id: Optional[str] = None,
            extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> AddResult:
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        chunks = [line.strip() for line in text.split("\n") if line.strip()]
        if not chunks:
            return AddResult(doc_id=doc_id, chunk_ids=[], chunks_added=0)

        ids = [f"{doc_id}::chunk-{i}" for i in range(len(chunks))]
        metadatas: List[Dict[str, Any]] = []
        for i in range(len(chunks)):
            md = {"doc_id": doc_id, "chunk_index": i, "source": source}
            if extra_metadata:
                md.update(extra_metadata)
            metadatas.append(md)

        self.collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return AddResult(doc_id=doc_id, chunk_ids=ids, chunks_added=len(chunks))

    def add_file(
        self,
        file_path: str,
        *,
        encoding: str = "utf-8",
        source: Optional[str] = None,
        doc_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> AddResult:
        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()
        return self.add_text(
            text,
            source=source or file_path,
            doc_id=doc_id,
            extra_metadata=extra_metadata,
        )

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[QueryHit]:
        res = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[QueryHit] = []
        for i in range(len(res["ids"][0])):
            hits.append(QueryHit(
                id=res["ids"][0][i],
                distance=float(res["distances"][0][i]),
                document=res["documents"][0][i],
                metadata=res["metadatas"][0][i],
            ))
        return hits


_CACHE: Dict[str, ChromaDB] = {}

def get_chroma_with_local_bge(
    *,
    model_path: str,
    persist_dir: str = "./chroma_db",
    collection_name: str = "documents",
    device: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 64,
) -> ChromaDB:
    key = "||".join([
        os.path.abspath(model_path),
        os.path.abspath(persist_dir),
        collection_name,
        str(device),
        str(normalize),
        str(batch_size),
    ])
    if key not in _CACHE:
        ef = BGEEmbeddingFunction(
            model_path=model_path,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
        )
        _CACHE[key] = ChromaDB(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_function=ef,
            space="cosine",
        )
    return _CACHE[key]