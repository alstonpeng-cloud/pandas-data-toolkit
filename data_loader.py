"""data_loader.py -- 通用表格数据载入与结构探查（基于 pandas）。

功能
----
1. 按扩展名自动分派读取 csv / excel(xlsx|xls) / parquet（parquet 需 pyarrow）。
2. 自动识别列语义类型：numeric / category / boolean / datetime / text，并统计空值数、唯一值数。
3. 抽样：按行数或比例，支持 random / head 两种方式，可固定 random_state。
4. 行数与内存保护：读取时限制行数（max_rows / nrows），载入后估算内存占用，超阈值只给提示而不中断。
5. 返回 DataFrame + 结构化信息（DatasetInfo），可直接序列化为 dict 供前端 / 小程序预览使用。

依赖: pandas, numpy；读取 parquet 需要 pyarrow；读取 excel 需要 openpyxl（xlsx）或 xlrd（xls）。

典型调用
--------
>>> result = read_data("orders.csv", max_rows=100_000, sample_rows=5_000, random_state=42)
>>> result.df.head()
>>> print(result.info.summary())
>>> preview = preview_data("orders.csv", n=5)   # 直接给上传预览用
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "ColumnKind",
    "SampleMethod",
    "ColumnInfo",
    "DatasetInfo",
    "LoadResult",
    "detect_format",
    "format_bytes",
    "estimate_memory_bytes",
    "infer_column_kind",
    "describe_columns",
    "read_data",
    "preview_data",
]

ColumnKind = Literal["numeric", "category", "boolean", "datetime", "text"]
SampleMethod = Literal["random", "head"]

# 各格式"文件体积 -> 内存占用"的经验膨胀系数（仅用于读取前的粗略提示）
_EXPANSION_FACTOR: dict[str, float] = {"csv": 2.0, "excel": 3.0, "parquet": 5.0}

_CSV_SUFFIXES = {".csv"}
_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
_PARQUET_SUFFIXES = {".parquet", ".pq"}


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def detect_format(path: str | os.PathLike[str]) -> str:
    """根据扩展名判断文件格式。

    Args:
        path: 文件路径。

    Returns:
        "csv" | "excel" | "parquet"。

    Raises:
        ValueError: 扩展名不受支持。
    """
    suffix = Path(path).suffix.lower()
    if suffix in _CSV_SUFFIXES:
        return "csv"
    if suffix in _EXCEL_SUFFIXES:
        return "excel"
    if suffix in _PARQUET_SUFFIXES:
        return "parquet"
    raise ValueError(
        f"不支持的文件扩展名 {suffix!r}，仅支持: .csv / .xlsx / .xls / .xlsm / .parquet / .pq"
    )


def format_bytes(size_bytes: int | float) -> str:
    """把字节数格式化为可读字符串，例如 ``0.42 MB``。"""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{int(round(size))} B" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"  # pragma: no cover - 兜底


def estimate_memory_bytes(
    path: str | os.PathLike[str], *, fmt: str | None = None
) -> int:
    """读取前基于文件体积粗略估算载入后的内存占用（字节）。

    这是一个经验值（csv ×2、excel ×3、parquet ×5），只用于给出"可能吃内存"的提示，
    不是精确测量；真实占用请用 ``DatasetInfo.memory_bytes``。
    """
    actual_fmt = fmt or detect_format(path)
    file_size = os.path.getsize(path)
    return int(file_size * _EXPANSION_FACTOR[actual_fmt])


def _frame_memory_bytes(df: pd.DataFrame) -> int:
    """统计 DataFrame 的实际内存占用（含 object 列的字符串内容）。"""
    return int(df.memory_usage(deep=True).sum())


def _to_python(value: Any) -> Any:
    """把 numpy / pandas 标量转成原生 Python 值，保证可 JSON 序列化。"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return str(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def infer_column_kind(
    series: pd.Series, *, category_max_unique: int = 50
) -> ColumnKind:
    """推断单列的语义类型。

    规则（按优先级）：
    1. bool dtype -> ``boolean``
    2. Categorical dtype -> ``category``
    3. datetime / timedelta dtype -> ``datetime``
    4. 数值 dtype -> ``numeric``
    5. 其余（object / string）：唯一值不超过 ``category_max_unique`` 记为 ``category``，否则记为 ``text``。

    Args:
        series: 待判断的列。
        category_max_unique: 低基数文本判定为类别的唯一值上限。
    """
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return "datetime"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    try:
        nunique = int(series.nunique(dropna=True))
    except TypeError:  # 元素不可哈希
        return "text"
    return "category" if nunique <= category_max_unique else "text"


@dataclass
class ColumnInfo:
    """单列的结构化信息。"""

    name: str
    dtype: str
    kind: ColumnKind
    non_null: int
    null_count: int
    null_ratio: float
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "kind": self.kind,
            "non_null": self.non_null,
            "null_count": self.null_count,
            "null_ratio": round(self.null_ratio, 4),
            "unique_count": self.unique_count,
            "sample_values": self.sample_values,
        }


def describe_columns(
    df: pd.DataFrame,
    *,
    category_max_unique: int = 50,
    sample_values: int = 3,
) -> list[ColumnInfo]:
    """生成每一列的类型 / 空值 / 唯一值 / 示例值信息。"""
    infos: list[ColumnInfo] = []
    n_rows = len(df)
    for name in df.columns:
        series = df[name]
        null_count = int(series.isna().sum())
        non_null = n_rows - null_count
        try:
            unique_count = int(series.nunique(dropna=True))
        except TypeError:
            unique_count = -1
        preview_values = [_to_python(v) for v in series.dropna().head(sample_values)]
        infos.append(
            ColumnInfo(
                name=str(name),
                dtype=str(series.dtype),
                kind=infer_column_kind(series, category_max_unique=category_max_unique),
                non_null=non_null,
                null_count=null_count,
                null_ratio=(null_count / n_rows) if n_rows else 0.0,
                unique_count=unique_count,
                sample_values=preview_values,
            )
        )
    return infos


@dataclass
class DatasetInfo:
    """一份数据集的基础信息，``to_dict()`` 可直接返回给前端。"""

    path: str | None
    fmt: str
    n_rows: int
    n_cols: int
    memory_bytes: int
    memory_human: str
    columns: list[ColumnInfo]
    sampling: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.fmt,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_bytes": self.memory_bytes,
            "memory_human": self.memory_human,
            "sampling": self.sampling,
            "warnings": self.warnings,
            "columns": [c.to_dict() for c in self.columns],
        }

    def summary(self) -> str:
        """生成可读的文本摘要。"""
        lines = [
            f"文件: {self.path or '<DataFrame>'} ({self.fmt})",
            f"形状: {self.n_rows} 行 x {self.n_cols} 列",
            f"内存: {self.memory_human} ({self.memory_bytes} bytes)",
        ]
        if self.sampling:
            sampling_desc = ", ".join(f"{k}={v}" for k, v in self.sampling.items() if v is not None)
            lines.append(f"抽样: {sampling_desc}")
        lines.append("列信息:")
        lines.append(f"  {'列名':<16}{'类型':<10}{'dtype':<14}{'缺失':>8}{'唯一值':>8}  示例")
        for col in self.columns:
            sample_text = ", ".join(str(v) for v in col.sample_values[:3])
            lines.append(
                f"  {col.name[:15]:<16}{col.kind:<10}{col.dtype:<14}"
                f"{col.null_count:>8}{col.unique_count:>8}  {sample_text[:40]}"
            )
        if self.warnings:
            lines.append("提示:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


@dataclass
class LoadResult:
    """``read_data`` 的返回值：数据本体 + 基础信息。"""

    df: pd.DataFrame
    info: DatasetInfo

    def preview(self, n: int = 5) -> dict[str, Any]:
        """就地生成前 n 行的预览结构（含列类型）。"""
        return {
            "shape": {"rows": self.info.n_rows, "cols": self.info.n_cols},
            "memory": {"bytes": self.info.memory_bytes, "human": self.info.memory_human},
            "columns": [c.to_dict() for c in self.info.columns],
            "rows": _rows_to_dicts(self.df.head(n)),
            "warnings": list(self.info.warnings),
        }


def _rows_to_dicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """把 DataFrame 转成逐行 dict，且所有值都是 JSON 友好的原生类型。"""
    return [
        {str(col): _to_python(value) for col, value in record.items()}
        for record in df.to_dict(orient="records")
    ]


# --------------------------------------------------------------------------- #
# 读取实现
# --------------------------------------------------------------------------- #
def _read_parquet(
    path: str | os.PathLike[str],
    limit: int | None,
    read_kwargs: Mapping[str, Any],
) -> pd.DataFrame:
    """读取 parquet；有行数上限时按 batch 读取，避免整表载入。"""
    try:
        import pyarrow.parquet as pq  # noqa: F401  (仅用于探测依赖)
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "读取 parquet 需要 pyarrow，请先执行: pip install pyarrow"
        ) from exc

    if limit is None:
        return pd.read_parquet(path, **read_kwargs)

    kwargs = dict(read_kwargs)
    columns = kwargs.pop("columns", None)
    if kwargs:  # filters 等复杂参数交给 pandas 处理，再截断
        return pd.read_parquet(path, columns=columns, **kwargs).head(limit)

    import pyarrow.parquet as pq

    if isinstance(columns, str):
        columns = [columns]
    parquet_file = pq.ParquetFile(path)
    batches = parquet_file.iter_batches(batch_size=max(int(limit), 1), columns=columns)
    try:
        batch = next(iter(batches))
    except StopIteration:  # 空文件
        return pd.read_parquet(path, columns=columns).head(0)
    return batch.to_pandas().head(limit).reset_index(drop=True)


def _apply_sampling(
    df: pd.DataFrame,
    *,
    sample_rows: int | None,
    sample_frac: float | None,
    method: SampleMethod,
    random_state: int | None,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按行数或比例抽样，返回抽样后的 DataFrame 与抽样描述。"""
    if sample_rows is None and sample_frac is None:
        return df, {}

    total = len(df)
    if sample_frac is not None:
        n = max(1, int(round(total * sample_frac)))
    else:
        n = int(sample_rows or 0)
        if n <= 0:
            raise ValueError("sample_rows 必须为正整数")
        if n > total:
            warnings.append(f"sample_rows={n} 超过可用行数 {total}，已取全部数据")
            n = total

    if method == "head":
        sampled = df.head(n)
    else:
        sampled = df.sample(n=n, random_state=random_state)

    return sampled, {
        "method": method,
        "sample_rows": sample_rows,
        "sample_frac": sample_frac,
        "random_state": random_state,
    }


def read_data(
    path: str | os.PathLike[str],
    *,
    max_rows: int | None = None,
    nrows: int | None = None,
    sample_rows: int | None = None,
    sample_frac: float | None = None,
    method: SampleMethod = "random",
    random_state: int | None = None,
    memory_limit_mb: float | None = None,
    category_max_unique: int = 50,
    read_kwargs: Mapping[str, Any] | None = None,
) -> LoadResult:
    """读取 csv / excel / parquet，并返回 DataFrame 与基础信息。

    读取顺序：先按行数上限截断（``max_rows`` / ``nrows``），再做抽样（``sample_rows`` / ``sample_frac``）。

    Args:
        path: 文件路径。
        max_rows: 最多读取的行数（与 ``nrows`` 等价，二者只能传一个）。
        nrows: ``max_rows`` 的别名，方便习惯 pandas 的用户。
        sample_rows: 抽样后的目标行数；与 ``sample_frac`` 互斥。
        sample_frac: 抽样比例，取值 (0, 1]；与 ``sample_rows`` 互斥。
        method: ``"random"`` 随机抽样（可复现）或 ``"head"`` 取前 N 行。
        random_state: 随机种子，method="random" 时保证结果可复现。
        memory_limit_mb: 内存阈值（MB）。超过时只在 ``info.warnings`` 里提示，不中断读取。
        category_max_unique: 低基数文本判定为类别的唯一值上限。
        read_kwargs: 透传给 pandas 读取函数的额外参数（如 ``sep``、``usecols``）。

    Returns:
        LoadResult，含 ``.df``（索引已重置）与 ``.info``。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 参数非法或扩展名不支持。
        ImportError: 读 parquet 缺 pyarrow，或读 excel 缺 openpyxl/xlrd。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if max_rows is not None and nrows is not None:
        raise ValueError("max_rows 与 nrows 含义相同，请只传其中一个")
    limit = max_rows if max_rows is not None else nrows
    if limit is not None and limit <= 0:
        raise ValueError("max_rows / nrows 必须为正整数")
    if sample_rows is not None and sample_frac is not None:
        raise ValueError("sample_rows 与 sample_frac 互斥，请只传其中一个")
    if sample_frac is not None and not 0 < sample_frac <= 1:
        raise ValueError("sample_frac 取值范围为 (0, 1]")
    if method not in ("random", "head"):
        raise ValueError("method 只能是 'random' 或 'head'")

    fmt = detect_format(file_path)
    kwargs = dict(read_kwargs or {})
    warnings: list[str] = []

    # 读取前的内存粗估：只提示，不阻断
    if memory_limit_mb is not None:
        estimated = estimate_memory_bytes(file_path, fmt=fmt)
        if estimated > memory_limit_mb * 1024 * 1024:
            warnings.append(
                f"按文件体积粗估载入后约需 {format_bytes(estimated)}，"
                f"超过阈值 {memory_limit_mb} MB；已通过行数限制/抽样降低占用"
            )

    if fmt == "csv":
        if limit is not None:
            kwargs.setdefault("nrows", int(limit))
        df = pd.read_csv(file_path, **kwargs)
    elif fmt == "excel":
        try:
            if limit is not None:
                kwargs.setdefault("nrows", int(limit))
            df = pd.read_excel(file_path, **kwargs)
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise ImportError(
                "读取 excel 需要 openpyxl（.xlsx）或 xlrd（.xls），请先安装对应依赖"
            ) from exc
    else:
        df = _read_parquet(file_path, limit, kwargs)
        if limit is not None and len(df) >= limit:
            warnings.append(f"parquet 已按前 {limit} 行读取")

    if limit is not None and len(df) < limit:
        warnings.append(f"文件实际只有 {len(df)} 行，未触及上限 {limit}")

    df, sampling = _apply_sampling(
        df,
        sample_rows=sample_rows,
        sample_frac=sample_frac,
        method=method,
        random_state=random_state,
        warnings=warnings,
    )
    df = df.reset_index(drop=True)

    memory_bytes = _frame_memory_bytes(df)
    if memory_limit_mb is not None and memory_bytes > memory_limit_mb * 1024 * 1024:
        warnings.append(
            f"当前数据占用 {format_bytes(memory_bytes)}，超过阈值 {memory_limit_mb} MB；"
            "建议调小 max_rows / sample_frac，或读取时指定 usecols / dtype"
        )

    info = DatasetInfo(
        path=str(file_path),
        fmt=fmt,
        n_rows=len(df),
        n_cols=df.shape[1],
        memory_bytes=memory_bytes,
        memory_human=format_bytes(memory_bytes),
        columns=describe_columns(df, category_max_unique=category_max_unique),
        sampling=sampling,
        warnings=warnings,
    )
    return LoadResult(df=df, info=info)


def preview_data(
    source: str | os.PathLike[str] | pd.DataFrame,
    n: int = 5,
    *,
    max_rows: int | None = 1000,
    memory_limit_mb: float | None = None,
    category_max_unique: int = 50,
) -> dict[str, Any]:
    """上传预览专用：返回前 n 行 + 每列类型 / 空值 / 唯一值的 dict。

    Args:
        source: 文件路径或已有 DataFrame。传路径时只读前 ``max_rows`` 行，避免上传大文件时卡死。
        n: 预览行数。
        max_rows: 传路径时的行数上限；传 None 表示不限制。
        memory_limit_mb: 内存阈值（MB），超过只在 ``warnings`` 里提示。
        category_max_unique: 低基数文本判定为类别的唯一值上限。

    Returns:
        形如 ``{"shape": ..., "memory": ..., "columns": [...], "rows": [...], "warnings": [...]}``。
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")

    if isinstance(source, pd.DataFrame):
        df = source
        warnings: list[str] = []
        memory_bytes = _frame_memory_bytes(df)
        if memory_limit_mb is not None and memory_bytes > memory_limit_mb * 1024 * 1024:
            warnings.append(
                f"当前数据占用 {format_bytes(memory_bytes)}，超过阈值 {memory_limit_mb} MB"
            )
        columns = describe_columns(df, category_max_unique=category_max_unique)
        return {
            "shape": {"rows": len(df), "cols": df.shape[1]},
            "memory": {"bytes": memory_bytes, "human": format_bytes(memory_bytes)},
            "columns": [c.to_dict() for c in columns],
            "rows": _rows_to_dicts(df.head(n)),
            "warnings": warnings,
        }

    result = read_data(
        source,
        max_rows=max_rows,
        memory_limit_mb=memory_limit_mb,
        category_max_unique=category_max_unique,
    )
    return {
        "shape": {"rows": result.info.n_rows, "cols": result.info.n_cols},
        "memory": {
            "bytes": result.info.memory_bytes,
            "human": result.info.memory_human,
        },
        "columns": [c.to_dict() for c in result.info.columns],
        "rows": _rows_to_dicts(result.df.head(n)),
        "warnings": result.info.warnings,
    }


# --------------------------------------------------------------------------- #
# 最小可运行示例
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(42)
    row_count = 600
    demo = pd.DataFrame(
        {
            "order_id": [f"A{i:05d}" for i in range(row_count)],
            "city": rng.choice(["北京", "上海", "深圳", "杭州"], row_count),
            "amount": np.round(rng.normal(200, 60, row_count), 2),
            "paid": rng.random(row_count) > 0.3,
            "created_at": pd.date_range("2024-01-01", periods=row_count, freq="h"),
            "note": [f"  备注 {i}  " for i in range(row_count)],
        }
    )
    demo.loc[[3, 7], "amount"] = np.nan

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = {
            "csv": Path(tmp_dir) / "orders.csv",
            "excel": Path(tmp_dir) / "orders.xlsx",
            "parquet": Path(tmp_dir) / "orders.parquet",
        }
        demo.to_csv(paths["csv"], index=False)
        demo.to_excel(paths["excel"], index=False)
        demo.to_parquet(paths["parquet"], index=False)

        for fmt_name, file_path in paths.items():
            print(f"\n{'=' * 60}\n读取 {fmt_name}: {file_path.name}")
            loaded = read_data(
                file_path,
                max_rows=300,
                sample_rows=80,
                random_state=7,
                memory_limit_mb=0.02,
            )
            print(loaded.info.summary())
            print(f"抽样后前 2 行:\n{loaded.df.head(2).to_string()}")

        print(f"\n{'=' * 60}\nhead 抽样 + 无抽样对比")
        head_sample = read_data(paths["csv"], sample_rows=3, method="head")
        print(head_sample.df[["order_id", "city"]])

        print(f"\n{'=' * 60}\npreview_data 预览（上传场景）")
        preview = preview_data(paths["csv"], n=3, max_rows=500)
        print("形状:", preview["shape"], "内存:", preview["memory"]["human"])
        print("首列信息:", preview["columns"][0])
        print("首行数据:", preview["rows"][0])
        print("提示:", preview["warnings"])
