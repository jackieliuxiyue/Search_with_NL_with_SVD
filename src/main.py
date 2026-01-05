from chroma_api_bge_local import ChromaDB, get_chroma_with_local_bge, AddResult
import os
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Set, Iterable, Callable
import re
from ragkit import RAGEngine, OpenAICompatibleChat

def read_csv_robust(path: str, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", "cp936"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="utf-8", errors="replace", **kwargs)

def _read_seen(txt_path: str) -> Set[str]:
    if not os.path.exists(txt_path):
        return set()
    with open(txt_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def _append_seen(txt_path: str, keys: List[str]) -> None:
    if not keys:
        return
    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
    with open(txt_path, "a", encoding="utf-8") as f:
        for k in keys:
            f.write(k + "\n")

def _seen_key_relpath(path: str, *, base_dir: str) -> str:
    return os.path.relpath(path, start=base_dir)

def csv_to_labeled_txt(
    csv_path: str,
    txt_path: str,
    *,
    encoding: str = "utf-8-sig",
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    row_sep: str = "\n\n",
    cell_sep: str = "；",
    kv_sep: str = "：",
    skip_null: bool = True,
) -> None:
    df = pd.read_csv(csv_path, encoding=encoding)

    cols = list(df.columns)
    if include is not None:
        include = list(include)
        cols = [c for c in include if c in df.columns]
    if exclude is not None:
        exclude_set = set(exclude)
        cols = [c for c in cols if c not in exclude_set]

    def row_to_text(row: pd.Series) -> str:
        parts = []
        for c in cols:
            v = row[c]
            if skip_null and (pd.isna(v) or v is None):
                continue
            s = str(v).strip()
            if skip_null and s == "":
                continue
            parts.append(f"{c}{kv_sep}{s}")
        return cell_sep.join(parts)

    with open(txt_path, "w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(df.iterrows()):
            line = row_to_text(row)
            if i > 0:
                f.write(row_sep)
            f.write(line)

def load_txt(path: str, encoding: str = "utf-8") -> List[Tuple[str, str]]:
    with open(path, "r", encoding=encoding) as f:
        text = f.read()

    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]

    docs: List[Tuple[str, str]] = [(str(i), blk) for i, blk in enumerate(blocks)]
    return docs

def add_docs_one_chunk_per_line(
    db: ChromaDB,
    docs: List[Tuple[str, str]],
    *,
    source: str = "table_doc",
    extra_metadata: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
    progress_every: int = 100,
) -> List[AddResult]:
    results: List[AddResult] = []
    total = len(docs)
    processed = 0
    added = 0

    for doc_id, content in docs:
        processed += 1

        text = (content or "").strip()
        if not text:
            results.append(AddResult(doc_id=doc_id, chunk_ids=[], chunks_added=0))
        else:
            chunk_id = f"{doc_id}::chunk-0"
            md = {
                "doc_id": doc_id,
                "chunk_index": 0,
                "source": source,
                "format": "full",
            }
            if extra_metadata:
                md.update(extra_metadata)

            db.collection.add(ids=[chunk_id], documents=[text], metadatas=[md])
            results.append(AddResult(doc_id=doc_id, chunk_ids=[chunk_id], chunks_added=1))
            added += 1

        if progress_cb and (processed % progress_every == 0 or processed == total):
            progress_cb(processed, total, added)

    return results

def print_progress(processed: int, total: int, added: int):
    print(f"[index] {processed}/{total} processed, {added} added")

def df_to_labeled_text(df, cell_sep="；", kv_sep="：", row_sep="\n"):

    lines = []

    for _, row in df.iterrows():
        parts = []

        for col in df.columns:
            v = row[col]

            if pd.isna(v):
                continue

            s = str(v).strip()

            if s == "":
                continue

            parts.append(f"{col}{kv_sep}{s}")
        lines.append(cell_sep.join(parts))

    return row_sep.join(lines)

def search_in_table(path: str, target: str, encoding: str = "utf-8") -> str:
    table = pd.read_csv(path, encoding = "utf-8-sig")
    col = "所属表/视图中文名"

    matched = table.loc[table[col] == target]
    text = df_to_labeled_text(matched, row_sep="\n")
    print(text)

    return text

def main():
    data_dir = "./data"

    csv_path = "./data/中台模型_资产导出_台账.csv"
    txt_path = "./data/中台模型_资产导出_台账.txt"
    csv_path_check = "./data/中台模型数据项.csv"

    seen_txt = "./data/seen_csv_index.txt"

    key = _seen_key_relpath(csv_path, base_dir=data_dir)
    seen = _read_seen(seen_txt)

    db = get_chroma_with_local_bge(
        model_path="./hf_models/bge-base-zh-v1.5",
        persist_dir="./chroma_db",
        collection_name="documents",
        device="cpu",
    )

    if key in seen:
        print(f"Skip (seen): {key}")
        docs = []
    else:
        csv_to_labeled_txt(csv_path, txt_path)
        docs = load_txt(txt_path)

    if docs:
        add_docs_one_chunk_per_line(
            db, docs,
            progress_cb=print_progress,
            progress_every=1
        )
        _append_seen(seen_txt, [key])
        print(f"Indexed {len(docs)} lines from {key}. Seen updated.")
    else:
        print("No lines to index.")

    llm = OpenAICompatibleChat()
    rag = RAGEngine(db=db, llm_chat=llm, prompt="你是数据字典助手。只能依据上下文回答。")

    prompt_1 = "你要协助一个数据分析师找到他需要的表格。请根据知识库返回的条目，找出问题需要用用到的的数据资产中文名。要求：只返回数据资产中文名，用换行符‘\n’隔开"

    while True:
        q = input("\n> ").strip()
        if not q or q.lower() in ("exit", "quit", "q"):
            break

        res = rag.ask(q, prompt=prompt_1, top_k=10, max_distance=0.45)
        print("\n数据分析助手>")
        print(res.answer)

        blocks = [b.strip() for b in res.answer.split("\n") if b.strip()]

        all_rows = []
        total = 0

        for i, block in enumerate(blocks):
            try:
                req = search_in_table(csv_path_check, block, encoding="utf-8")
            except Exception as e:
                print(e)
                continue

            all_rows.append(req)
            total += len(req)

        if not all_rows:
            continue

        merged = "\n".join(all_rows)
        print(merged)

        rows_text = (
                merged
                + "\n"
                + "以上是知识库检索出的相关的行，请根据这些知识，帮助一个数据分析员，"
                  "他想要分析一个具体的问题，你需要帮助他找到他所需要的数据资产，或者生成相应的分析方法"
        )

        res_2 = rag.ask_without_search(q, prompt=rows_text)
        print(res_2.answer)

if __name__ == "__main__":
    main()













