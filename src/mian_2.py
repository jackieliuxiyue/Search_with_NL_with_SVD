from chroma_api_bge_local import ChromaDB, get_chroma_with_local_bge, AddResult
import os
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Set, Iterable, Callable, Union, Sequence
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

def search_in_table(
    path: str,
    target: Union[Sequence[str], Tuple[str, str]],
    encoding: str = "utf-8",
) -> str:
    table = pd.read_csv(path, encoding="utf-8-sig")

    asset_col = "资产中文名称"
    view_col = "所属表/视图中文名"

    if not isinstance(target, (list, tuple)) or len(target) != 2:
        raise ValueError("target 必须是长度为2的 [资产中文名称, 所属表/视图中文名]")

    asset_name, view_name = target[0], target[1]

    matched = table.loc[
        (table[asset_col].astype(str).str.strip() == str(asset_name).strip())
        & (table[view_col].astype(str).str.strip() == str(view_name).strip())
    ]

    text = df_to_labeled_text(matched, row_sep="\n")
    print(text)
    return text

def search_in_table_certain_row(
    path: str,
    target: str,
    encoding: str = "utf-8",
    cols_keep: Optional[List[str]] = None
) -> str:
    table = pd.read_csv(path, encoding="utf-8-sig")
    key_col = "所属表/视图中文名"

    matched = table.loc[table[key_col] == target]

    if cols_keep is not None:
        cols_keep = [c for c in cols_keep if c in matched.columns]
        matched = matched.loc[:, cols_keep]

    text = df_to_labeled_text(matched, row_sep="\n")
    print(text)
    return text

def line_to_tuple(line: str) -> Tuple[str, str]:
    m = re.fullmatch(r"\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]", line.strip())
    if not m:
        raise ValueError(f"格式不符合: {line!r}，期望形如 [资产中文名, 所属表/视图中文名]")
    return (m.group(1).strip(), m.group(2).strip())

def main():
    global final_answer
    data_dir = "./data"

    csv_path = "./data/中台模型_资产导出_台账.csv"
    txt_path = "./data/中台模型_资产导出_台账.txt"
    csv_path_check = "./data/中台模型数据项_台账.csv"

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
        print("---------------------------------------------------------------------------------")

        blocks = [b.strip() for b in res.answer.split("\n") if b.strip()]

        all_rows = []
        total_2 = 0

        need_cols = ["资产中文名称", "资产定义说明", "所属表/视图中文名"]

        for i, block in enumerate(blocks):
            try:
                req = search_in_table_certain_row(csv_path_check, block, cols_keep=need_cols)
                prompt_each = (
                    req
                    + "\n"
                    + "以上是知识库检索出的相关的数据资产，请根据数据分析员想要分析的具体的问题。要求：只返回你认为有用的资产。要求：返回格式为[资产中文名,所属表/视图中文名]，不要展示列名，不要有其他，每一条用‘\n’分隔"
                )

                res_2 = rag.ask_without_search(q, prompt=prompt_each)
                print(res_2.answer)
                print("---------------------------------------------------------------------------------")

                blocks_2 = [b.strip() for b in res_2.answer.split("\n") if b.strip()]

                all_rows_2 = []
                total = 0

                for j, block_2 in enumerate(blocks_2):
                    try:
                        k = line_to_tuple(block_2)
                        req_2 = search_in_table(csv_path_check, k)
                    except Exception as e:
                        print(e)
                        continue

                    all_rows_2.append(req_2)
                    total += len(req_2)

                merged = "\n".join(all_rows_2)
                all_rows.append(merged)
                total_2 += len(req)
                print("---------------------------------------------------------------------------------")

            except Exception as e:
                print(e)
                continue

        if not all_rows:
            continue

        print("---------------------------------------------------------------------------------")
        merged_2 = "\n".join(all_rows)
        print(merged_2)
        print("---------------------------------------------------------------------------------")

        rows_text = (
                merged_2
                + "\n"
                + "以上是知识库检索出的相关的数据资产，请根据数据分析员想要分析的具体的问题。"
                + "他最初的问题是："
                + q
        )

        res_3 = rag.ask_without_search(q, prompt=rows_text)
        print(res_3.answer)
        print("---------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()

