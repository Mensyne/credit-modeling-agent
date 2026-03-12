#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交叉特征生成管道 - Screen守护进程优化版
用法: screen -dmS crossfeat python cross_feature_pipeline.py --config config.xlsx --data data.parquet --output ./results/
"""

import pandas as pd
import numpy as np
import logging
import string
import re
import sys
import os
import argparse
import signal
import atexit
from typing import Optional, Tuple, Dict, List
from itertools import product, combinations
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# ============================================================================
# 配置与常量
# ============================================================================

DEFAULT_BINS = 10
LOG_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# ============================================================================
# 日志配置
# ============================================================================

def setup_logger(log_file: Optional[str] = None, 
                 log_level: str = 'INFO',
                 console_output: bool = True) -> logging.Logger:
    """
    配置日志系统 - 支持文件+控制台双输出

    Parameters:
    -----------
    log_file : str, optional
        日志文件路径，为None时只输出到控制台
    log_level : str
        日志级别: DEBUG, INFO, WARNING, ERROR
    console_output : bool
        是否同时输出到控制台（screen模式下建议True）
    """
    logger = logging.getLogger('CrossFeaturePipeline')
    logger.setLevel(getattr(logging, log_level.upper()))

    # 清除已有handlers避免重复
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 控制台Handler - 带颜色（如果在终端）
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件Handler
    if log_file:
        # 确保目录存在
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# 进程管理工具
# ============================================================================

class ProcessManager:
    """进程状态管理器 - 处理信号和优雅退出"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.running = True
        self.start_time = datetime.now()
        self.stats = {
            'processed_combinations': 0,
            'failed_combinations': 0,
            'skipped_combinations': 0
        }

        # 注册信号处理
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # 注册退出回调
        atexit.register(self._on_exit)

    def _handle_signal(self, signum, frame):
        """处理终止信号"""
        sig_name = 'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'
        self.logger.warning(f"\n收到 {sig_name} 信号，准备优雅退出...")
        self.running = False

    def _on_exit(self):
        """进程退出时输出统计"""
        duration = datetime.now() - self.start_time
        self.logger.info("=" * 80)
        self.logger.info("进程即将退出")
        self.logger.info(f"运行时长: {duration}")
        self.logger.info(f"成功处理: {self.stats['processed_combinations']} 个组合")
        self.logger.info(f"失败: {self.stats['failed_combinations']} 个组合")
        self.logger.info(f"跳过: {self.stats['skipped_combinations']} 个组合")
        self.logger.info("=" * 80)

    def check_running(self) -> bool:
        """检查是否应该继续运行"""
        return self.running

    def update_stat(self, key: str, increment: int = 1):
        """更新统计"""
        self.stats[key] += increment


# ============================================================================
# 核心功能函数
# ============================================================================

def generate_labels(n: int) -> List[str]:
    """生成字母标签序列: A-Z, AA, AB, ..."""
    labels = []
    chars = string.ascii_uppercase

    for i in range(n):
        if i < 26:
            labels.append(chars[i])
        else:
            i -= 26
            first = i // 26
            second = i % 26
            labels.append(chars[first] + chars[second])

    return labels


def parse_score_range(range_str: str) -> Tuple[float, float]:
    """解析分数区间字符串 [min,max] 或 [min-max]"""
    clean_str = range_str.strip().replace('[', '').replace(']', '').replace(' ', '')

    if ',' in clean_str:
        parts = clean_str.split(',')
    elif '-' in clean_str:
        parts = clean_str.split('-')
    else:
        raise ValueError(f"无法解析分数区间: {range_str}")

    if len(parts) != 2:
        raise ValueError(f"分数区间格式错误: {range_str}")

    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        raise ValueError(f"分数区间包含非数字值: {range_str}")


def parse_trend(trend_str: str) -> str:
    """解析趋势字符串为 positive/negative"""
    trend_str = trend_str.strip()
    if '风险越高' in trend_str:
        return 'negative'
    elif '风险越低' in trend_str:
        return 'positive'
    else:
        raise ValueError(f"无法识别趋势类型: {trend_str}")


def load_score_config(file_path: str, logger: logging.Logger) -> pd.DataFrame:
    """加载分数产品配置文件"""
    logger.info(f"正在加载配置文件: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件不存在: {file_path}")

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("仅支持 .xlsx 和 .csv 文件")

    required_cols = ['分数产品', '趋势', '分数区间']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"配置文件缺少必需列: {missing}")

    df['趋势_parsed'] = df['趋势'].apply(parse_trend)
    df[['分数区间_min', '分数区间_max']] = df['分数区间'].apply(
        lambda x: pd.Series(parse_score_range(x))
    )

    logger.info(f"成功加载 {len(df)} 个分数产品配置")
    return df


def create_cross_feature(
    df: pd.DataFrame,
    score_col1: str,
    score_col2: str,
    bins_num: int,
    score_col1_range: Optional[Tuple[float, float]] = None,
    score_col2_range: Optional[Tuple[float, float]] = None,
    trend_col1: str = 'positive',
    trend_col2: str = 'positive',
    na_policy: str = 'nan',
    verbose_mapping: bool = False,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    创建两个分数列的交叉分箱特征
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 参数验证
    if trend_col1 not in ['positive', 'negative']:
        raise ValueError("trend_col1 必须是 'positive' 或 'negative'")
    if trend_col2 not in ['positive', 'negative']:
        raise ValueError("trend_col2 必须是 'positive' 或 'negative'")
    if score_col1 not in df.columns:
        raise ValueError(f"列 '{score_col1}' 不存在")
    if score_col2 not in df.columns:
        raise ValueError(f"列 '{score_col2}' 不存在")
    if na_policy not in ['nan', 'unknown']:
        raise ValueError("na_policy 必须是 'nan' 或 'unknown'")
    if bins_num < 1:
        raise ValueError("bins_num 必须 >= 1")

    new_fea_col = f"{score_col1}_and_{score_col2}"
    new_fea_desc_col = f"{new_fea_col}_desc"

    logger.info(f"处理: {score_col1} x {score_col2}, 分箱数: {bins_num}")

    df = df.copy()
    missing_mask = df[score_col1].isna() | df[score_col2].isna()
    needs_reverse = (trend_col1 != trend_col2)

    if needs_reverse:
        logger.info(f"  趋势相反，{score_col2} 将反转")

    # 确定范围
    min1, max1 = score_col1_range if score_col1_range else (df[score_col1].min(), df[score_col1].max())
    min2, max2 = score_col2_range if score_col2_range else (df[score_col2].min(), df[score_col2].max())

    if pd.isna(min1) or pd.isna(max1):
        raise ValueError(f"{score_col1} 范围无效，请提供 score_col1_range")
    if pd.isna(min2) or pd.isna(max2):
        raise ValueError(f"{score_col2} 范围无效，请提供 score_col2_range")

    # 归一化
    denom1 = max1 - min1
    denom2 = max2 - min2

    df[f'{score_col1}_norm'] = 0.0 if denom1 == 0 else ((df[score_col1] - min1) / denom1).clip(0, 1)
    raw_norm2 = 0.0 if denom2 == 0 else ((df[score_col2] - min2) / denom2).clip(0, 1)
    df[f'{score_col2}_norm'] = 1 - raw_norm2 if needs_reverse else raw_norm2

    # 分箱
    norm_bins = np.linspace(0, 1, bins_num + 1)

    if bins_num == 1:
        df[f'{score_col1}_bin_idx'] = 0
        df[f'{score_col2}_bin_idx'] = 0
    else:
        df[f'{score_col1}_bin_idx'] = pd.cut(
            df[f'{score_col1}_norm'], bins=norm_bins, include_lowest=True, right=True, labels=False
        ).astype('Int64')
        df[f'{score_col2}_bin_idx'] = pd.cut(
            df[f'{score_col2}_norm'], bins=norm_bins, include_lowest=True, right=True, labels=False
        ).astype('Int64')

    # 生成映射
    labels = generate_labels(bins_num ** 2)
    bin_indices = list(product(range(bins_num), range(bins_num)))

    mapping_dict = {}
    mapping_desc = {}

    for (i, j), label in zip(bin_indices, labels):
        mapping_dict[(i, j)] = label

        orig_low1 = min1 + (max1 - min1) * norm_bins[i]
        orig_high1 = min1 + (max1 - min1) * norm_bins[i + 1]

        if needs_reverse:
            orig_high2 = min2 + (max2 - min2) * (1 - norm_bins[j])
            orig_low2 = min2 + (max2 - min2) * (1 - norm_bins[j + 1])
        else:
            orig_low2 = min2 + (max2 - min2) * norm_bins[j]
            orig_high2 = min2 + (max2 - min2) * norm_bins[j + 1]

        desc1 = f"{orig_low1:.4f}<={score_col1}<={orig_high1:.4f}" if i == 0 else f"{orig_low1:.4f}<{score_col1}<={orig_high1:.4f}"
        desc2 = f"{orig_low2:.4f}<={score_col2}<={orig_high2:.4f}" if j == 0 else f"{orig_low2:.4f}<{score_col2}<={orig_high2:.4f}"
        mapping_desc[(i, j)] = f"{desc1}_and_{desc2}"

    # 应用映射
    combo_idx = df[f'{score_col1}_bin_idx'] * bins_num + df[f'{score_col2}_bin_idx']
    label_map = {i * bins_num + j: mapping_dict[(i, j)] for i, j in bin_indices}
    desc_map = {i * bins_num + j: mapping_desc[(i, j)] for i, j in bin_indices}

    df[new_fea_col] = combo_idx.map(label_map)
    df[new_fea_desc_col] = combo_idx.map(desc_map)

    # 缺失值处理
    if na_policy == 'unknown':
        df.loc[missing_mask, [new_fea_col, new_fea_desc_col]] = 'Unknown'
    else:
        df.loc[missing_mask, [new_fea_col, new_fea_desc_col]] = np.nan

    if verbose_mapping:
        for key in sorted(mapping_desc.keys()):
            logger.info(f"  {mapping_dict[key]}: {mapping_desc[key]}")

    # 添加分箱区间列
    df[f'{score_col1}_norm_bin'] = pd.cut(df[f'{score_col1}_norm'], bins=norm_bins, include_lowest=True, right=True)
    df[f'{score_col2}_norm_bin'] = pd.cut(df[f'{score_col2}_norm'], bins=norm_bins, include_lowest=True, right=True)

    # 清理
    df = df.drop(columns=[f'{score_col1}_bin_idx', f'{score_col2}_bin_idx'])

    logger.info(f"  完成: 新增 {new_fea_col}, {new_fea_desc_col}")
    return df


def generate_combinations(score_list: List[str], 
                         n_pairs: Optional[int] = None, 
                         random_seed: int = 42) -> List[Tuple[str, str]]:
    """生成分数产品的两两组合"""
    all_combos = list(combinations(score_list, 2))

    if n_pairs and n_pairs < len(all_combos):
        import random
        random.seed(random_seed)
        return random.sample(all_combos, n_pairs)
    return all_combos


def process_cross_features(
    data_df: pd.DataFrame,
    config_df: pd.DataFrame,
    bins_num: int = 10,
    combinations_list: Optional[List[Tuple[str, str]]] = None,
    verbose_mapping: bool = False,
    pm: Optional[ProcessManager] = None,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """批量处理交叉特征"""

    if logger is None:
        logger = logging.getLogger(__name__)

    if combinations_list is None:
        combinations_list = generate_combinations(config_df['分数产品'].tolist())

    total = len(combinations_list)
    logger.info(f"\n{'='*80}")
    logger.info(f"开始处理 {total} 个组合")
    logger.info(f"{'='*80}\n")

    # 构建配置字典
    config_dict = {}
    for _, row in config_df.iterrows():
        config_dict[row['分数产品']] = {
            'trend': row['趋势_parsed'],
            'range': (row['分数区间_min'], row['分数区间_max'])
        }

    result_df = data_df

    for idx, (col1, col2) in enumerate(combinations_list, 1):
        # 检查是否应该停止
        if pm and not pm.check_running():
            logger.warning(f"进程被中断，已处理 {idx-1}/{total} 个组合")
            break

        logger.info(f"[{idx}/{total}] {col1} x {col2}")

        # 检查配置和数据
        cfg1, cfg2 = config_dict.get(col1), config_dict.get(col2)
        if not cfg1 or not cfg2:
            logger.warning(f"  跳过: 配置缺失")
            if pm: pm.update_stat('skipped_combinations')
            continue
        if col1 not in result_df.columns or col2 not in result_df.columns:
            logger.warning(f"  跳过: 数据列缺失")
            if pm: pm.update_stat('skipped_combinations')
            continue

        try:
            result_df = create_cross_feature(
                df=result_df,
                score_col1=col1,
                score_col2=col2,
                bins_num=bins_num,
                score_col1_range=cfg1['range'],
                score_col2_range=cfg2['range'],
                trend_col1=cfg1['trend'],
                trend_col2=cfg2['trend'],
                verbose_mapping=verbose_mapping,
                logger=logger
            )
            if pm: pm.update_stat('processed_combinations')

        except Exception as e:
            logger.error(f"  错误: {str(e)}")
            if pm: pm.update_stat('failed_combinations')
            # 可以选择在这里中断或继续
            # raise  # 如果需要失败即停，取消注释

    logger.info(f"\n{'='*80}")
    logger.info(f"处理完成: {result_df.shape}")
    logger.info(f"{'='*80}")
    return result_df


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='交叉特征生成管道 - Screen守护进程版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础用法（模拟数据）
  python cross_feature_pipeline.py --config config.xlsx

  # 使用真实数据
  python cross_feature_pipeline.py --config config.xlsx --data data.parquet --output ./results/

  # Screen守护进程模式
  screen -dmS crossfeat python cross_feature_pipeline.py --config config.xlsx --data data.parquet --log ./logs/run.log

  # 指定分箱数和随机采样组合
  python cross_feature_pipeline.py --config config.xlsx --data data.parquet --bins 5 --sample-pairs 100
        """
    )

    parser.add_argument('--config', '-c', required=True,
                       help='分数产品配置文件路径 (.xlsx 或 .csv)')
    parser.add_argument('--data', '-d', default=None,
                       help='输入数据文件路径 (.parquet, .csv, .feather)，不指定则生成模拟数据')
    parser.add_argument('--output', '-o', default='./output/',
                       help='输出目录 (默认: ./output/)')
    parser.add_argument('--log', '-l', default=None,
                       help='日志文件路径，默认输出到控制台')
    parser.add_argument('--bins', '-b', type=int, default=10,
                       help='分箱数量 (默认: 10)')
    parser.add_argument('--sample-pairs', '-s', type=int, default=None,
                       help='随机采样组合数，默认处理全部组合')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--verbose-mapping', action='store_true',
                       help='输出详细的分箱映射关系')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')

    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    log_file = args.log or output_dir / f"cross_feature_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = setup_logger(
        log_file=str(log_file),
        log_level=args.log_level,
        console_output=True  # Screen模式下保持控制台输出
    )

    # 初始化进程管理器
    pm = ProcessManager(logger)

    logger.info("=" * 80)
    logger.info("交叉特征生成管道启动")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"输出目录: {output_dir.absolute()}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 80)

    try:
        # 1. 加载配置
        config_df = load_score_config(args.config, logger)

        # 2. 加载/生成数据
        if args.data:
            logger.info(f"\n加载数据: {args.data}")
            if args.data.endswith('.parquet'):
                data_df = pd.read_parquet(args.data)
            elif args.data.endswith('.csv'):
                data_df = pd.read_csv(args.data)
            elif args.data.endswith('.feather'):
                data_df = pd.read_feather(args.data)
            else:
                raise ValueError(f"不支持的文件格式: {args.data}")
            logger.info(f"数据形状: {data_df.shape}")
        else:
            logger.info("\n生成模拟数据...")
            np.random.seed(args.seed)
            n_samples = 10000
            data_dict = {'user_id': range(n_samples)}
            for _, row in config_df.iterrows():
                min_v, max_v = row['分数区间_min'], row['分数区间_max']
                data_dict[row['分数产品']] = np.random.uniform(min_v, max_v, n_samples)
            data_df = pd.DataFrame(data_dict)
            logger.info(f"模拟数据形状: {data_df.shape}")

        # 3. 生成组合列表
        score_products = config_df['分数产品'].tolist()
        if args.sample_pairs:
            combinations_list = generate_combinations(score_products, args.sample_pairs, args.seed)
            logger.info(f"\n随机采样 {args.sample_pairs} 个组合")
        else:
            combinations_list = generate_combinations(score_products)
            logger.info(f"\n处理全部 {len(combinations_list)} 个组合")

        # 4. 批量处理
        result_df = process_cross_features(
            data_df=data_df,
            config_df=config_df,
            bins_num=args.bins,
            combinations_list=combinations_list,
            verbose_mapping=args.verbose_mapping,
            pm=pm,
            logger=logger
        )

        # 5. 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"cross_features_{timestamp}.parquet"
        result_df.to_parquet(output_file, index=False)
        logger.info(f"\n结果已保存: {output_file}")

        # 6. 统计
        orig_cols = set(data_df.columns)
        new_cols = [c for c in result_df.columns if c not in orig_cols]
        logger.info(f"\n统计:")
        logger.info(f"  原始列数: {len(orig_cols)}")
        logger.info(f"  新增列数: {len(new_cols)}")
        logger.info(f"  总列数: {len(result_df.columns)}")

        # 保存列信息
        meta_file = output_dir / f"cross_features_{timestamp}_meta.txt"
        with open(meta_file, 'w', encoding='utf-8') as f:
            f.write(f"原始列 ({len(orig_cols)}):\n")
            for c in sorted(orig_cols):
                f.write(f"  {c}\n")
            f.write(f"\n新增列 ({len(new_cols)}):\n")
            for c in sorted(new_cols):
                f.write(f"  {c}\n")
        logger.info(f"  元数据保存: {meta_file}")

        logger.info("\n" + "=" * 80)
        logger.info("处理成功完成!")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.exception("程序执行失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())