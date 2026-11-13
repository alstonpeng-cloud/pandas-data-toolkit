"""data_cleaner.py -- 轻量数据清洗工具（基于 pandas）。

功能
----
1. 缺失值：删除行 / 删除列、均值 / 中位数 / 众数填充，可按列指定不同策略；
   非数值列遇到均值/中位数会自动退回众数（或跳过）并写入日志。
2. 重复值：整行或指定 subset 去重。
3. 异常值：IQR 与 Z-score 两种检测，支持"仅标记"与"删除"两种模式，阈值可配。
4. 类型转换：to_numeric / to_datetime / category / string / boolean。
5. 日期解析：支持自定义格式（字符串或按列指定）与容错（coerce / raise）。
6. 字符串去首尾空格。

每次操作都会写入结构化日志（列名、操作、影响条数/比例、前后行数），
可通过 ``summary()`` 拿到可打印摘要、``log_frame()`` 拿到 DataFrame、``to_dict()`` 拿到 dict。

依赖: pandas, numpy。无全局状态、无硬编码路径，两个模块互不依赖。

典型调用
--------
>>> cleaner = (
...     DataCleaner(df)
...     .strip_whitespace()
...     .fill_missing({"age": "median", "city": "mode"})
...     .drop_duplicates()
...     .handle_outliers(method="iqr", columns=["age"], mode="flag")
...     .convert_types({"user_id": "int", "signup": "datetime"})
... )
>>> clean_df = cleaner.result()
>>> print(cleaner.summary())
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["CleanLogEntry", "DataCleaner", "OutlierMethod", "FillStrategy"]

OutlierMethod = Literal["iqr", "zscore"]
FillStrategy = Literal["mean", "median", "mode", "drop", "skip"]

_VALID_FILL: set[str] = {"mean", "median", "mode", "drop", "skip"}
_TRUE_TOKENS = {"true", "1", "yes", "y", "t", "是", "真", "有"}
_FALSE_TOKENS = {"false", "0", "no", "n", "f", "否", "假", "无"}


@dataclass
class CleanLogEntry:
    """单条清洗日志。"""

    step: int
    operation: str
    column: str | None
    affected: int
    ratio: float
    rows_before: int
    rows_after: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ratio"] = round(self.ratio, 4)
        return data


def _to_bool(value: Any) -> Any:
    """把常见布尔写法转成 True / False / pd.NA。"""
    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) if value in (0, 1) else pd.NA
    text = str(value).strip().lower()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    return pd.NA


def _fmt_scalar(value: Any) -> str:
    """日志里的数值展示：浮点保留 4 位。"""
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    return str(value)


def _outlier_mask(
    series: pd.Series, method: OutlierMethod, threshold: float
) -> pd.Series | None:
    """计算单列异常值掩码；无法计算（全空 / 方差为 0）时返回 None。"""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return None

    if method == "iqr":
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            return None
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    else:  # zscore
        mean = values.mean()
        std = values.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return None
        z_scores = (values - mean) / std
        mask = z_scores.abs() > threshold
        return mask.fillna(False)

    mask = (values < lower) | (values > upper)
    return mask.fillna(False)


class DataCleaner:
    """链式调用的轻量清洗器。

    内部维护一份 DataFrame 副本（``copy=False`` 时直接操作传入对象）与操作日志。
    所有清洗方法都返回 ``self``，便于链式调用；通过 ``result()`` 取结果、``summary()`` 看摘要。
    """

    def __init__(self, df: pd.DataFrame, *, copy: bool = True) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas.DataFrame")
        self._df = df.copy(deep=True) if copy else df
        self._log: list[CleanLogEntry] = []
        self._step = 0

    # ------------------------------------------------------------------ #
    # 结果与日志
    # ------------------------------------------------------------------ #
    @property
    def df(self) -> pd.DataFrame:
        """当前（清洗中）的 DataFrame。"""
        return self._df

    @property
    def log(self) -> list[CleanLogEntry]:
        """清洗日志列表（只读副本）。"""
        return list(self._log)

    def result(self) -> pd.DataFrame:
        """返回清洗后的 DataFrame。"""
        return self._df

    def log_frame(self) -> pd.DataFrame:
        """日志转为 DataFrame，便于导出 csv / 展示。"""
        columns = [
            "step",
            "operation",
            "column",
            "affected",
            "ratio",
            "rows_before",
            "rows_after",
            "detail",
        ]
        if not self._log:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([entry.to_dict() for entry in self._log], columns=columns)

    def to_dict(self) -> dict[str, Any]:
        """结构化结果：形状 + 每步日志 + 文本摘要。"""
        return {
            "shape": {"rows": len(self._df), "cols": self._df.shape[1]},
            "steps": [entry.to_dict() for entry in self._log],
            "summary": self.summary(),
        }

    def summary(self) -> str:
        """可打印的清洗摘要。"""
        lines = [
            "清洗摘要",
            "=" * 8,
            f"当前形状: {len(self._df)} 行 x {self._df.shape[1]} 列",
        ]
        if not self._log:
            lines.append("尚未执行任何清洗操作")
            return "\n".join(lines)
        lines.append(f"共 {len(self._log)} 步操作:")
        for entry in self._log:
            column = entry.column or "-"
            lines.append(
                f"  [{entry.step:>2}] {entry.operation:<24} 列={column:<18}"
                f" 影响={entry.affected:<6} 比例={entry.ratio:6.2%}"
                f" 行数 {entry.rows_before}->{entry.rows_after}"
                + (f" | {entry.detail}" if entry.detail else "")
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 内部记录
    # ------------------------------------------------------------------ #
    def _record(
        self,
        operation: str,
        column: str | None,
        affected: int,
        rows_before: int,
        detail: str = "",
        *,
        denominator: int | None = None,
    ) -> None:
        """记录一条日志；``denominator`` 默认为当前行数（如按列统计时可传列数）。"""
        base = rows_before if denominator is None else denominator
        self._step += 1
        self._log.append(
            CleanLogEntry(
                step=self._step,
                operation=operation,
                column=column,
                affected=int(affected),
                ratio=(affected / base) if base else 0.0,
                rows_before=int(rows_before),
                rows_after=int(len(self._df)),
                detail=detail,
            )
        )

    def _require_columns(self, columns: Sequence[str]) -> None:
        missing = [c for c in columns if c not in self._df.columns]
        if missing:
            raise KeyError(f"列不存在: {missing}")

    def _numeric_columns(self) -> list[str]:
        return [
            str(c)
            for c in self._df.columns
            if pd.api.types.is_numeric_dtype(self._df[c].dtype)
            and not pd.api.types.is_bool_dtype(self._df[c].dtype)
        ]

    # ------------------------------------------------------------------ #
    # 缺失值
    # ------------------------------------------------------------------ #
    def drop_missing(
        self,
        *,
        axis: Literal["rows", "columns"] = "rows",
        subset: Sequence[str] | None = None,
        how: Literal["any", "all"] = "any",
        thresh: int | None = None,
    ) -> "DataCleaner":
        """删除含缺失值的行或列。

        Args:
            axis: ``"rows"`` 删行，``"columns"`` 删列。
            subset: 仅在这些列上判断缺失（axis="rows" 时有效）。
            how: ``"any"`` 任一为空即删，``"all"`` 全空才删。
            thresh: 非空值数量下限，少于该值才删除（与 how 互斥使用时以 thresh 为准）。

        Returns:
            self
        """
        if axis not in ("rows", "columns"):
            raise ValueError("axis 只能是 'rows' 或 'columns'")
        if subset is not None:
            subset = list(subset)
            self._require_columns(subset)

        # pandas 不允许同时传 how 与 thresh
        drop_kwargs: dict[str, Any] = {"axis": 0 if axis == "rows" else 1, "subset": subset}
        if thresh is not None:
            drop_kwargs["thresh"] = thresh
        else:
            drop_kwargs["how"] = how

        if axis == "rows":
            rows_before = len(self._df)
            self._df = self._df.dropna(**drop_kwargs)
            detail = (f"thresh={thresh}" if thresh is not None else f"how={how}") + (
                f", subset={subset}" if subset else ""
            )
            self._record("drop_missing_rows", None, rows_before - len(self._df), rows_before, detail)
        else:
            cols_before = self._df.shape[1]
            before_columns = list(self._df.columns)
            self._df = self._df.dropna(**drop_kwargs)
            dropped = [c for c in before_columns if c not in set(self._df.columns)]
            self._record(
                "drop_missing_columns",
                None,
                len(dropped),
                len(self._df),
                f"删除列: {dropped}" if dropped else "无列被删除",
                denominator=cols_before,
            )
        return self

    def fill_missing(
        self,
        strategy: str | Mapping[str, str] = "median",
        *,
        columns: Sequence[str] | None = None,
        non_numeric_fallback: Literal["mode", "skip"] = "mode",
    ) -> "DataCleaner":
        """填充（或按列删除）缺失值。

        Args:
            strategy: 统一策略字符串，或 ``{列名: 策略}`` 的映射。策略取值：
                ``mean`` / ``median`` / ``mode`` / ``drop``（删除该列为空的行）/ ``skip``。
            columns: 当 strategy 为字符串时，指定作用列；默认作用于所有存在缺失值的列。
            non_numeric_fallback: 非数值列使用 mean/median 时的兜底策略：``"mode"``（默认）或 ``"skip"``。

        Returns:
            self
        """
        if non_numeric_fallback not in ("mode", "skip"):
            raise ValueError("non_numeric_fallback 只能是 'mode' 或 'skip'")

        if isinstance(strategy, Mapping):
            plan = {str(col): str(s).lower() for col, s in strategy.items()}
            if columns is not None:
                allowed = set(columns)
                unknown = [c for c in plan if c not in allowed]
                if unknown:
                    raise ValueError(f"以下列不在 columns 范围内: {unknown}")
        else:
            key = str(strategy).lower()
            targets = list(columns) if columns is not None else [
                str(c) for c in self._df.columns if self._df[c].isna().any()
            ]
            plan = {col: key for col in targets}

        for col, rule in plan.items():
            if col not in self._df.columns:
                raise KeyError(f"列不存在: {col!r}")
            if rule not in _VALID_FILL:
                raise ValueError(f"列 {col!r} 的策略 {rule!r} 非法，可选: {sorted(_VALID_FILL)}")

        for col, rule in plan.items():
            series = self._df[col]
            na_mask = series.isna()
            rows_before = len(self._df)

            if not na_mask.any():
                self._record("fill_missing", col, 0, rows_before, "无缺失值，跳过")
                continue
            if rule == "skip":
                self._record("fill_missing", col, 0, rows_before, "按要求跳过")
                continue
            if rule == "drop":
                self._df = self._df.loc[~na_mask]
                self._record(
                    "fill_missing:drop_rows",
                    col,
                    int(na_mask.sum()),
                    rows_before,
                    f"删除 {col} 为空的行",
                )
                continue

            effective_rule = rule
            note = ""
            is_numeric = pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
                series.dtype
            )
            if rule in ("mean", "median") and not is_numeric:
                if non_numeric_fallback == "skip":
                    self._record(
                        "fill_missing",
                        col,
                        int(na_mask.sum()),
                        rows_before,
                        f"{rule} 不适用于非数值 dtype={series.dtype}，按要求跳过",
                    )
                    continue
                effective_rule = "mode"
                note = f"dtype={series.dtype} 非数值，{rule} 已退回 mode"

            if effective_rule in ("mean", "median"):
                value = series.mean() if effective_rule == "mean" else series.median()
            else:  # mode
                modes = series.mode(dropna=True)
                value = modes.iloc[0] if len(modes) else None

            if value is None or (isinstance(value, float) and math.isnan(value)):
                self._record(
                    "fill_missing", col, int(na_mask.sum()), rows_before, "无法计算填充值，跳过"
                )
                continue

            filled = series.fillna(value)
            self._df[col] = filled
            detail = f"{effective_rule} -> {_fmt_scalar(value)}"
            if note:
                detail = f"{detail}（{note}）"
            self._record("fill_missing", col, int(na_mask.sum()), rows_before, detail)

        return self

    # ------------------------------------------------------------------ #
    # 重复值
    # ------------------------------------------------------------------ #
    def drop_duplicates(
        self,
        *,
        subset: Sequence[str] | None = None,
        keep: Literal["first", "last", False] = "first",
    ) -> "DataCleaner":
        """删除重复行。

        Args:
            subset: 参与判重的列，默认整行。
            keep: 保留 ``"first"`` / ``"last"``，或 ``False`` 表示全部删除。

        Returns:
            self
        """
        if subset is not None:
            subset = list(subset)
            self._require_columns(subset)
        rows_before = len(self._df)
        dup_mask = self._df.duplicated(subset=subset, keep=keep)
        duplicate_count = int(dup_mask.sum())
        self._df = self._df.loc[~dup_mask]
        detail = f"keep={keep}" + (f", subset={subset}" if subset else ", 整行判重")
        self._record("drop_duplicates", None, rows_before - len(self._df), rows_before, f"{detail}, 重复 {duplicate_count} 行")
        return self

    # ------------------------------------------------------------------ #
    # 异常值
    # ------------------------------------------------------------------ #
    def handle_outliers(
        self,
        method: OutlierMethod = "iqr",
        *,
        columns: Sequence[str] | None = None,
        threshold: float | None = None,
        mode: Literal["flag", "drop"] = "flag",
        flag_column: str = "outlier_flag",
    ) -> "DataCleaner":
        """异常值检测：仅标记或直接删除。

        Args:
            method: ``"iqr"``（默认阈值 1.5 倍 IQR）或 ``"zscore"``（默认 |z| > 3）。
            columns: 参与检测的数值列，默认所有数值列。
            threshold: IQR 时为倍数 k；Z-score 时为 |z| 阈值。必须 > 0。
            mode: ``"flag"`` 新增布尔列标记（任一列异常即为 True），``"drop"`` 删除异常行。
            flag_column: mode="flag" 时新增的列名。

        Returns:
            self
        """
        if method not in ("iqr", "zscore"):
            raise ValueError("method 只能是 'iqr' 或 'zscore'")
        if mode not in ("flag", "drop"):
            raise ValueError("mode 只能是 'flag' 或 'drop'")

        default_threshold = 1.5 if method == "iqr" else 3.0
        k = default_threshold if threshold is None else float(threshold)
        if k <= 0:
            raise ValueError("threshold 必须为正数")

        if columns is None:
            target_columns = self._numeric_columns()
        else:
            target_columns = [str(c) for c in columns]
            self._require_columns(target_columns)

        rows_before = len(self._df)
        combined = pd.Series(False, index=self._df.index)
        counts: dict[str, int] = {}
        skipped: list[str] = []

        for col in target_columns:
            if not pd.api.types.is_numeric_dtype(self._df[col].dtype) or pd.api.types.is_bool_dtype(
                self._df[col].dtype
            ):
                skipped.append(f"{col}(非数值)")
                continue
            mask = _outlier_mask(self._df[col], method, k)
            if mask is None:
                skipped.append(f"{col}(无法计算)")
                continue
            counts[col] = int(mask.sum())
            combined = combined | mask

        if not counts:
            self._record(
                "handle_outliers",
                None,
                0,
                rows_before,
                f"method={method}, threshold={k}, 无可检测列; 跳过: {skipped}",
            )
            return self

        detail = (
            f"method={method}, threshold={k}, 各列异常: {counts}"
            + (f", 跳过: {skipped}" if skipped else "")
        )

        if mode == "flag":
            self._df[flag_column] = combined
            self._record("handle_outliers:flag", flag_column, int(combined.sum()), rows_before, detail)
        else:
            self._df = self._df.loc[~combined]
            self._record(
                "handle_outliers:drop", None, rows_before - len(self._df), rows_before, detail
            )
        return self

    # ------------------------------------------------------------------ #
    # 类型转换
    # ------------------------------------------------------------------ #
    def convert_types(
        self,
        mapping: Mapping[str, str],
        *,
        errors: Literal["coerce", "raise"] = "coerce",
    ) -> "DataCleaner":
        """批量类型转换。

        Args:
            mapping: ``{列名: 目标类型}``，目标类型支持
                ``numeric`` / ``int`` / ``float`` / ``datetime`` / ``category`` / ``string`` / ``bool``。
            errors: 转换失败时的行为，``"coerce"`` 置空（默认），``"raise"`` 抛错。

        Returns:
            self
        """
        if errors not in ("coerce", "raise"):
            raise ValueError("errors 只能是 'coerce' 或 'raise'")
        if not mapping:
            return self
        self._require_columns(list(mapping.keys()))

        for col, target in mapping.items():
            series = self._df[col]
            before_na = int(series.isna().sum())
            rows_before = len(self._df)
            key = str(target).strip().lower()
            source_dtype = str(series.dtype)

            if key in ("numeric", "number", "float", "float64"):
                converted = pd.to_numeric(series, errors=errors)
                if key in ("float", "float64"):
                    converted = converted.astype("float64")
            elif key in ("int", "integer", "int64"):
                converted = pd.to_numeric(series, errors=errors).astype("Int64")
            elif key in ("datetime", "date", "datetime64"):
                converted = pd.to_datetime(series, errors=errors)
            elif key == "category":
                converted = series.astype("category")
            elif key in ("string", "str", "text"):
                converted = series.astype("string")
            elif key in ("bool", "boolean"):
                converted = series.map(_to_bool).astype("boolean")
            else:
                raise ValueError(
                    f"列 {col!r} 的目标类型 {target!r} 不支持；"
                    "可选: numeric / int / float / datetime / category / string / bool"
                )

            after_na = int(converted.isna().sum())
            failed = max(0, after_na - before_na)
            self._df[col] = converted
            self._record(
                "convert_types",
                col,
                len(converted),
                rows_before,
                f"{source_dtype} -> {converted.dtype}, 失败 {failed} 个值"
                + ("（已置空）" if failed and errors == "coerce" else ""),
            )
        return self

    def parse_dates(
        self,
        columns: str | Sequence[str],
        *,
        fmt: str | Mapping[str, str] | None = None,
        errors: Literal["coerce", "raise"] = "coerce",
    ) -> "DataCleaner":
        """日期解析，支持自定义格式与容错。

        Args:
            columns: 单个列名或列名列表。
            fmt: 日期格式字符串（如 ``"%Y-%m-%d"``），或 ``{列名: 格式}``；默认自动推断。
            errors: ``"coerce"`` 无法解析的值置为 NaT（默认），``"raise"`` 直接抛错。

        Returns:
            self
        """
        if errors not in ("coerce", "raise"):
            raise ValueError("errors 只能是 'coerce' 或 'raise'")
        cols = [columns] if isinstance(columns, str) else list(columns)
        self._require_columns(cols)

        for col in cols:
            series = self._df[col]
            before_na = int(series.isna().sum())
            rows_before = len(self._df)
            column_fmt = fmt.get(col) if isinstance(fmt, Mapping) else fmt
            note = ""

            if column_fmt is None and not pd.api.types.is_datetime64_any_dtype(series.dtype):
                # 先按推断格式整体解析；混合格式（如 2024-01-02 与 2024/03/04）导致失败时，
                # 回退到逐元素解析（format="mixed"，pandas >= 2.0），避免容错模式下误丢日期。
                converted = pd.to_datetime(series, errors=errors)
                failed_auto = max(0, int(converted.isna().sum()) - before_na)
                if failed_auto > 0 and errors == "coerce":
                    try:
                        mixed = pd.to_datetime(series, errors="coerce", format="mixed")
                    except (TypeError, ValueError):  # pandas < 2.0
                        mixed = converted
                    if max(0, int(mixed.isna().sum()) - before_na) < failed_auto:
                        converted = mixed
                        note = "推断格式失败，已回退逐元素解析"
            else:
                converted = pd.to_datetime(series, format=column_fmt, errors=errors)

            failed = max(0, int(converted.isna().sum()) - before_na)
            self._df[col] = converted
            self._record(
                "parse_dates",
                col,
                len(converted),
                rows_before,
                f"format={column_fmt or 'infer'}, {str(series.dtype)} -> {converted.dtype}, "
                f"失败 {failed} 个值" + ("（已置空）" if failed else "") + (f"；{note}" if note else ""),
            )
        return self

    # ------------------------------------------------------------------ #
    # 字符串
    # ------------------------------------------------------------------ #
    def strip_whitespace(
        self,
        columns: Sequence[str] | None = None,
        *,
        side: Literal["both", "left", "right"] = "both",
    ) -> "DataCleaner":
        """去除字符串列的首尾空格。

        Args:
            columns: 指定列；默认所有 object / string 列。
            side: ``"both"`` / ``"left"`` / ``"right"``。

        Returns:
            self
        """
        if side not in ("both", "left", "right"):
            raise ValueError("side 只能是 'both' / 'left' / 'right'")

        if columns is None:
            cols = [
                str(c)
                for c in self._df.columns
                if pd.api.types.is_object_dtype(self._df[c].dtype)
                or pd.api.types.is_string_dtype(self._df[c].dtype)
            ]
        else:
            cols = [str(c) for c in columns]
            self._require_columns(cols)

        for col in cols:
            series = self._df[col]
            rows_before = len(self._df)
            if not (
                pd.api.types.is_object_dtype(series.dtype)
                or pd.api.types.is_string_dtype(series.dtype)
            ):
                self._record(
                    "strip_whitespace", col, 0, rows_before, f"dtype={series.dtype} 非字符串，跳过"
                )
                continue

            if isinstance(series.dtype, pd.StringDtype):
                stripped = series.str.strip() if side == "both" else (
                    series.str.lstrip() if side == "left" else series.str.rstrip()
                )
            else:
                def _strip(value: Any) -> Any:
                    if not isinstance(value, str):
                        return value
                    if side == "both":
                        return value.strip()
                    return value.lstrip() if side == "left" else value.rstrip()

                stripped = series.map(_strip)

            changed_mask = (series.ne(stripped) & series.notna() & stripped.notna()).fillna(False)
            changed = int(changed_mask.sum())
            self._df[col] = stripped
            self._record(
                "strip_whitespace",
                col,
                changed,
                rows_before,
                f"去除{ {'both': '首尾', 'left': '左侧', 'right': '右侧'}[side] }空格",
            )
        return self


# --------------------------------------------------------------------------- #
# 最小可运行示例
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    raw = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
            "city": [" 北京", "上海 ", " 深圳", None, "杭州", "北京", "上海", "深圳", "杭州", None, "北京"],
            "age": [23, 25, np.nan, 120, -5, 31, 28, np.nan, 45, 33, 33],
            "score": [88.5, 92.0, 79.5, np.nan, 85.0, 300.0, 91.0, 77.5, 88.0, 84.5, 84.5],
            "signup": [
                "2024-01-02", "2024-02-03", "2024/03/04", "bad-date", "2024-05-06",
                None, "2024-07-08", "2024-08-09", "2024-09-10", "2024-10-11", "2024-10-11",
            ],
            "active": ["true", "false", "TRUE", "no", "yes", "1", "0",
                       "yes", "no", "true", "true"],
        }
    )
    print("原始数据:")
    print(raw, "\n")

    cleaner = (
        DataCleaner(raw)
        .strip_whitespace()
        .fill_missing({"age": "median", "city": "mode", "score": "median"})
        .drop_duplicates(subset=["user_id"])
        .handle_outliers(method="iqr", columns=["age", "score"], threshold=1.5, mode="flag")
        .handle_outliers(method="zscore", columns=["score"], threshold=2.0, mode="drop")
        .parse_dates("signup", errors="coerce")
        .convert_types({"user_id": "int", "active": "bool", "city": "category"})
    )

    print(cleaner.summary())
    print("\n清洗日志 DataFrame:")
    print(cleaner.log_frame().to_string(index=False))
    print("\n清洗结果:")
    print(cleaner.result().to_string(index=False))
    print("\n结构化结果(dict)键:", list(cleaner.to_dict().keys()))
