import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


def _detect_encoding(csv_path: str, encodings=("utf-8-sig", "utf-8", "gbk", "gb2312", "cp936")) -> str:
    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                f.read(4096)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"

def _read_header_and_desc(csv_path: str, enc: str) -> Tuple[List[str], List[str]]:
    with open(csv_path, "r", encoding=enc, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        desc = next(reader, [])  # 第二行：列介绍（可能不存在）
    cols = [str(c).strip() for c in header]
    descs = [("" if d is None else str(d).strip()) for d in desc]
    return cols, descs

def _infer_scalar_type(values: List[str]) -> str:
    xs = [v.strip() for v in values if v is not None and str(v).strip() != ""]
    if not xs:
        return "empty"

    bool_set = {"true", "false", "0", "1", "yes", "no", "y", "n", "是", "否"}
    if all(x.lower() in bool_set for x in xs):
        return "bool"

    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except Exception:
            return False

    if all(is_int(x) for x in xs):
        return "int"

    def is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    if all(is_float(x) for x in xs):
        return "float"

    date_fmts = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"]
    dt_fmts = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"]

    def match_any(s: str, fmts: List[str]) -> bool:
        for fmt in fmts:
            try:
                datetime.strptime(s, fmt)
                return True
            except Exception:
                pass
        return False

    if all(match_any(x, dt_fmts) for x in xs):
        return "datetime"
    if all(match_any(x, date_fmts) for x in xs):
        return "date"

    return "string"


def csv_profile_to_string(
    csv_path: str,
    *,
    base_dir: Optional[str] = None,
    sample_rows: int = 5,
    max_cell_chars: int = 60,
) -> str:
    table_name = os.path.basename(csv_path)
    path = os.path.relpath(csv_path, start=base_dir) if base_dir else csv_path
    fmt = "csv"
    enc = _detect_encoding(csv_path)

    cols, descs = _read_header_and_desc(csv_path, enc)

    # 用列名个数对齐描述长度
    if len(descs) < len(cols):
        descs = descs + [""] * (len(cols) - len(descs))
    if len(descs) > len(cols):
        descs = descs[:len(cols)]

    # 从第三行开始取样（跳过列名+介绍）
    rows: List[List[str]] = []
    with open(csv_path, "r", encoding=enc, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        next(reader, None)  # desc row
        for _ in range(sample_rows):
            try:
                r = next(reader)
            except StopIteration:
                break
            rows.append([("" if v is None else str(v)) for v in r])

    col_samples: Dict[str, List[str]] = {c: [] for c in cols if str(c).strip() != ""}
    for r in rows:
        for i, c in enumerate(cols):
            c = str(c).strip()
            if not c:
                continue
            if i < len(r):
                v = r[i].strip()
                if v != "":
                    v = v[:max_cell_chars]
                    col_samples[c].append(v)

    col_types: Dict[str, str] = {c: _infer_scalar_type(col_samples.get(c, [])) for c in col_samples.keys()}

    lines = []
    lines.append("TABLE_DESC")
    lines.append(f"name: {table_name}")
    lines.append(f"path: data/{path}")
    lines.append(f"format: {fmt}")
    lines.append(f"encoding: {enc}")
    lines.append(f"columns_count: {len([c for c in cols if str(c).strip()])}")
    lines.append("columns:")

    for i, c in enumerate(cols):
        c = str(c).strip()
        if not c:
            continue
        d = descs[i] if i < len(descs) else ""
        samples = col_samples.get(c, [])[:3]
        samples_str = ", ".join([repr(s) for s in samples]) if samples else "(no sample)"
        type_guess = col_types.get(c, "empty")

        # 新增：desc
        if d:
            lines.append(f"  - {c} | desc: {d} | type_guess: {type_guess} | samples: {samples_str}")
        else:
            lines.append(f"  - {c} | type_guess: {type_guess} | samples: {samples_str}")

    return "\n".join(lines)

def csv_profile_to_min_text(csv_path: str, *, base_dir: Optional[str] = None, max_desc_cols: int = 80) -> str:
    table_name = os.path.basename(csv_path)
    rel = os.path.relpath(csv_path, start=base_dir) if base_dir else csv_path
    path = rel.replace("\\", "/")
    enc = _detect_encoding(csv_path)

    cols, descs = _read_header_and_desc(csv_path, enc)

    # 只保留非空列名
    pairs = []
    for i, c in enumerate(cols):
        c = str(c).strip()
        if not c:
            continue
        d = descs[i].strip() if i < len(descs) and descs[i] is not None else ""
        pairs.append((c, d))

    cols_str = "、".join([c for c, _ in pairs])

    # 新增：把“列:介绍”也塞进 min_text（可控长度）
    desc_items = []
    for c, d in pairs[:max_desc_cols]:
        if d:
            desc_items.append(f"{c}({d})")
        else:
            desc_items.append(c)
    desc_str = "、".join(desc_items)
    
    print(f"TABLE {table_name} | path=data/{path} | enc={enc} | cols={cols_str} | desc={desc_str}")

    return f"TABLE {table_name} | path=data/{path} | enc={enc} | cols={cols_str} | desc={desc_str}"