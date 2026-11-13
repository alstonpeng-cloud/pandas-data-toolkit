# pandas-data-toolkit

Pandas-based data loading and cleaning toolkit: read CSV/Excel/Parquet with type inspection, sampling and memory guards, then clean missing values, duplicates, outliers and dtypes with a structured log.

两个可独立 import 的轻量 Python 模块，面向"上传文件 → 预览 → 清洗 → 导出"这类常见数据流水线：

| 文件 | 作用 | 是否依赖另一个模块 |
| --- | --- | --- |
| `data_loader.py` | 按扩展名读取 csv / excel / parquet，识别列类型、抽样、行数与内存保护、上传预览 | 否 |
| `data_cleaner.py` | 缺失值、重复值、异常值、类型转换、日期解析、字符串去空格，并输出结构化清洗日志 | 否 |

两个模块都只依赖 `pandas` + `numpy`，无全局状态、无硬编码路径、无副作用（`if __name__ == "__main__"` 里是可运行的最小示例）。

## 安装

要求 Python >= 3.9。

```bash
git clone https://github.com/<your-name>/pandas-data-toolkit.git
cd pandas-data-toolkit
pip install -r requirements.txt
```

只装必需依赖（csv 场景够用）：

```bash
pip install pandas numpy
```

`requirements.txt` 中已包含读取 parquet 用的 `pyarrow` 和读取 `.xlsx` 用的 `openpyxl`；
读 `.xls` 还需 `pip install xlrd`。

直接用 GitHub 压缩包也可以：下载后解压，把 `data_loader.py` / `data_cleaner.py` 放到你的项目里 import 即可，无包结构要求。

## 快速开始

```bash
python data_loader.py     # 演示三种格式读取、抽样、内存提示、上传预览
python data_cleaner.py    # 演示完整清洗流水线与日志输出
```

---

## 一、`data_loader.py`

### 主要接口

| 函数 / 类 | 用途 |
| --- | --- |
| `read_data(path, ...) -> LoadResult` | 读取文件，返回 `.df` 和 `.info`；自动按扩展名分派格式 |
| `preview_data(source, n=5, ...) -> dict` | 上传预览：前 N 行 + 每列类型/空值/唯一值，返回值可直接 JSON 序列化 |
| `describe_columns(df) -> list[ColumnInfo]` | 逐列统计类型、空值数、空值率、唯一值数、示例值 |
| `infer_column_kind(series)` | 单列语义类型推断：`numeric` / `category` / `boolean` / `datetime` / `text` |
| `detect_format(path)` | 按扩展名返回 `csv` / `excel` / `parquet`，不支持时抛 `ValueError` |
| `format_bytes(n)` / `estimate_memory_bytes(path)` | 可读内存字符串 / 读取前的内存粗估 |
| `DatasetInfo.summary()` | 可直接 `print` 的文本摘要（文件、形状、内存、列信息、提示） |
| `DatasetInfo.to_dict()` | 结构化信息，方便丢给前端 / 小程序 |

`LoadResult` 字段：`df: DataFrame`（索引已重置）、`info: DatasetInfo`，另有 `result.preview(n=5)` 便捷方法。

### 典型调用

```python
from data_loader import read_data, preview_data

# 1) 常规读取 + 基础信息
result = read_data("orders.csv")
print(result.info.summary())
print(result.info.n_rows, result.info.memory_human)

# 2) 只读前 10 万行，再随机抽 5000 行（固定种子可复现）
result = read_data(
    "orders.csv",
    max_rows=100_000,        # 与 nrows=100_000 等价，二者只能传一个
    sample_rows=5_000,       # 或 sample_frac=0.1
    method="random",         # "head" 表示直接取前 N 行
    random_state=42,
    memory_limit_mb=50,      # 超阈值只在 info.warnings 里提示，不会中断
)
df = result.df

# 3) 读取参数透传（如自定义分隔符、只取部分列）
result = read_data("data.csv", read_kwargs={"sep": ";", "usecols": ["id", "amount"]})

# 4) 上传预览（小程序 / Web 场景）
preview = preview_data("upload.xlsx", n=5, max_rows=1000)
# {
#   "shape": {"rows": 1000, "cols": 6},
#   "memory": {"bytes": ..., "human": "140.54 KB"},
#   "columns": [{"name": "amount", "kind": "numeric", "dtype": "float64",
#                "null_count": 3, "null_ratio": 0.003, "unique_count": 997,
#                "sample_values": [...]}, ...],
#   "rows": [{"order_id": "A00001", "amount": 182.47, ...}],
#   "warnings": []
# }
```

### 行为约定

- **格式分派**：`.csv` / `.xlsx` / `.xls` / `.xlsm` / `.parquet` / `.pq`，其它扩展名直接报错。
- **列类型**：pandas dtype 优先；object 列唯一值 ≤ `category_max_unique`（默认 50）判为 `category`，否则 `text`。
  CSV 中的日期列默认是字符串（`text`），需要显式解析：
  `read_data("x.csv", read_kwargs={"parse_dates": ["created_at"]})`，或交给 `data_cleaner.parse_dates`。
- **抽样**：先按 `max_rows` / `nrows` 截断，再按 `sample_rows` / `sample_frac` 抽样；
  `method="random"` + `random_state` 可复现，`method="head"` 取前 N 行。
- **行数保护**：csv / excel 通过 `nrows` 限制读取；parquet 用 `pyarrow` 按 batch 只读前 N 行，不整表载入。
- **内存保护**：读取前按文件体积粗估（csv ×2 / excel ×3 / parquet ×5），读取后用
  `memory_usage(deep=True)` 实测；超过 `memory_limit_mb` 时只写入 `info.warnings`，不会抛异常。
- **异常**：文件不存在 → `FileNotFoundError`；参数冲突/非法 → `ValueError`；缺 pyarrow / openpyxl → `ImportError`。

---

## 二、`data_cleaner.py`

### 主要接口

`DataCleaner(df, copy=True)` 为链式调用，所有方法返回 `self`；结果与日志通过属性/方法读取。

| 方法 | 说明 |
| --- | --- |
| `drop_missing(axis="rows", subset=None, how="any", thresh=None)` | 删除含缺失值的行或列 |
| `fill_missing(strategy="median", columns=None, non_numeric_fallback="mode")` | 填充缺失值；`strategy` 可传统一字符串或 `{列名: 策略}` |
| `drop_duplicates(subset=None, keep="first")` | 去重，可按 subset 列判重 |
| `handle_outliers(method="iqr", columns=None, threshold=None, mode="flag", flag_column="outlier_flag")` | 异常值检测：`flag` 仅标记 / `drop` 删除 |
| `convert_types(mapping, errors="coerce")` | 类型转换：`numeric` / `int` / `float` / `datetime` / `category` / `string` / `bool` |
| `parse_dates(columns, fmt=None, errors="coerce")` | 日期解析，支持自定义格式（字符串或按列映射）与容错 |
| `strip_whitespace(columns=None, side="both")` | 字符串去首尾空格 |
| `result()` / `df` | 取清洗后的 DataFrame |
| `log` / `log_frame()` | 结构化日志（列表 / DataFrame） |
| `summary()` | 可打印的多行摘要 |
| `to_dict()` | `{"shape": ..., "steps": [...], "summary": ...}` |

`fill_missing` 支持策略：`mean` / `median` / `mode` / `drop`（删除该列为空的行）/ `skip`。
非数值列使用 `mean` / `median` 时，按 `non_numeric_fallback` 自动退回 `mode` 或跳过，并在日志 detail 中注明。

`handle_outliers` 阈值：IQR 默认 `1.5`（倍数 k，越界为 `Q1-k·IQR` / `Q3+k·IQR`），
Z-score 默认 `3.0`（`|z| > threshold`）。`mode="flag"` 会新增一列布尔标记（任一检测列异常即为 `True`），
`mode="drop"` 直接删除异常行。

### 典型调用

```python
from data_cleaner import DataCleaner

cleaner = (
    DataCleaner(raw_df)
    .strip_whitespace()                                        # 去首尾空格
    .fill_missing({"age": "median", "city": "mode"})           # 按列指定策略
    .drop_missing(axis="columns", thresh=100)                  # 非空值太少的列直接删
    .drop_duplicates(subset=["user_id"])                       # 按业务主键去重
    .handle_outliers(method="iqr", columns=["age", "amount"],
                     threshold=1.5, mode="flag")               # 只标记，保留数据
    .parse_dates("signup", errors="coerce")                    # 容错解析日期
    .convert_types({"user_id": "int", "city": "category",
                    "active": "bool"})                         # 类型收敛
)

clean_df = cleaner.result()
print(cleaner.summary())          # 人读摘要
print(cleaner.log_frame())        # 结构化日志（可导出 csv）
payload = cleaner.to_dict()       # 给前端 / 接口返回
```

### 日志字段

每步操作记录一条，字段如下：

| 字段 | 含义 |
| --- | --- |
| `step` | 步骤序号 |
| `operation` | 操作名，如 `fill_missing`、`handle_outliers:drop` |
| `column` | 涉及列名（整行操作时为 `None`） |
| `affected` | 影响条数（删行=删除行数，填充=填充单元格数，类型转换=整列行数） |
| `ratio` | `affected / 当前行数`（删列时以列数为分母） |
| `rows_before` / `rows_after` | 操作前后的行数 |
| `detail` | 补充说明：填充值、阈值、各列异常数、转换失败数等 |

---

## 设计说明

- **纯函数 + 轻量类**：`data_loader` 以函数为主（`read_data` / `preview_data`），
  `data_cleaner` 用 `DataCleaner` 保存日志状态；两者互不 import，可分别使用。
- **无隐式全局状态**：随机性只来自显式传入的 `random_state`，输出只取决于入参与数据。
- **类型注解 + docstring**：全部公开 API 带类型注解和中文 docstring。
- **不修改入参**：`DataCleaner(copy=True)` 默认深拷贝，原始 DataFrame 不会被改动。
- **可序列化**：预览与日志中的 numpy / pandas 标量统一转成原生 Python 类型，可直接 `json.dumps`。

## 已知限制

- `read_data` 对 parquet 的 `filters` 等复杂参数会走 pandas 整表读取后再截断；仅 `columns` 支持按批限量读取。
- Excel 依赖 openpyxl 解析，超大 xlsx 本身加载就慢，建议先转 csv / parquet。
- 列类型识别基于 dtype 和唯一值基数，语义判断（如"手机号是类别还是文本"）仍需业务侧确认。

## License

本项目基于 [MIT License](LICENSE) 开源，Copyright (c) 2026 alstonpeng-cloud。
