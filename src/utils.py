import pandas as pd

input_csv = "data/中台模型_资产导出.csv"
output_csv = "data/中台模型_资产导出_台账.csv"

df = pd.read_csv(input_csv, encoding="utf-8-sig")

col = "模式名"
if col not in df.columns:
    raise KeyError(f"未找到列：{col}，实际列为：{list(df.columns)}")

mask = df[df[col].isin(["CDS", "AADM"])]

mask.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"已导出 {len(mask)} 行到 {output_csv}")