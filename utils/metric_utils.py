import polars as pl
import numpy as np
from typing import List, Tuple


def calculate_psi(
    expected: pl.Series,
    actual: pl.Series,
    bins: int = 10,
    bin_type: str = "equal_width",
) -> Tuple[float, pl.DataFrame]:
    """
    计算PSI（群体稳定性指数）
    :param expected: 期望分布（训练集）
    :param actual: 实际分布（验证/测试/OOT集）
    :param bins: 分箱数
    :param bin_type: 分箱类型：equal_width等宽, equal_freq等频
    :return: (psi值, 分箱明细)
    """
    # 合并数据计算分箱边界
    all_data = pl.concat([expected, actual])

    if bin_type == "equal_width":
        bins = np.linspace(all_data.min(), all_data.max(), bins + 1)
    elif bin_type == "equal_freq":
        bins = all_data.quantile([i / bins for i in range(bins + 1)]).to_list()
    else:
        raise ValueError(f"不支持的分箱类型：{bin_type}")

    # 计算各分箱占比
    expected_counts = (
        expected.hist(bins=bins)
        .select("break_point", "count")
        .rename({"count": "expected_count"})
    )
    actual_counts = (
        actual.hist(bins=bins)
        .select("break_point", "count")
        .rename({"count": "actual_count"})
    )

    psi_df = expected_counts.join(actual_counts, on="break_point", how="left")
    psi_df = psi_df.with_columns(
        [
            (pl.col("expected_count") / expected.count()).alias("expected_pct"),
            (pl.col("actual_count") / actual.count()).alias("actual_pct"),
        ]
    )

    # 避免除以0和log(0)
    psi_df = psi_df.with_columns(
        [
            pl.when(pl.col("expected_pct") == 0)
            .then(1e-10)
            .otherwise(pl.col("expected_pct"))
            .alias("expected_pct"),
            pl.when(pl.col("actual_pct") == 0)
            .then(1e-10)
            .otherwise(pl.col("actual_pct"))
            .alias("actual_pct"),
        ]
    )

    psi_df = psi_df.with_columns(
        (
            (pl.col("actual_pct") - pl.col("expected_pct"))
            * np.log(pl.col("actual_pct") / pl.col("expected_pct"))
        ).alias("psi")
    )

    total_psi = psi_df["psi"].sum()
    return total_psi, psi_df


def calculate_iv(df: pl.DataFrame, feature_col: str, label_col: str) -> float:
    """计算IV值（信息价值）"""
    # 统计各分组的好坏样本数
    group = df.group_by(feature_col).agg(
        [
            pl.col(label_col).sum().alias("bad"),
            (pl.col(label_col).count() - pl.col(label_col).sum()).alias("good"),
        ]
    )

    total_bad = df[label_col].sum()
    total_good = df[label_col].count() - total_bad

    # 避免除以0
    group = group.with_columns(
        [
            pl.when(pl.col("bad") == 0).then(1).otherwise(pl.col("bad")).alias("bad"),
            pl.when(pl.col("good") == 0)
            .then(1)
            .otherwise(pl.col("good"))
            .alias("good"),
        ]
    )

    group = group.with_columns(
        [
            (pl.col("bad") / total_bad).alias("bad_pct"),
            (pl.col("good") / total_good).alias("good_pct"),
        ]
    )

    group = group.with_columns(
        (pl.col("bad_pct") - pl.col("good_pct"))
        * np.log(pl.col("bad_pct") / pl.col("good_pct"))
    ).alias("iv")

    return group["iv"].sum()


def get_feature_stability_level(psi: float) -> str:
    """获取特征稳定性等级"""
    if psi < 0.1:
        return "✅ 稳定"
    elif psi < 0.25:
        return "⚠️ 轻微漂移"
    else:
        return "❌ 严重漂移"
