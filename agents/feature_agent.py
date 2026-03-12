import streamlit as st
import polars as pl
import toad
import plotly.express as px
from utils.metric_utils import calculate_psi, calculate_iv, get_feature_stability_level


def render_feature_agent():
    st.title("🔧 特征工程Agent")

    if "train_data" not in st.session_state:
        st.warning("请先完成样本划分！")
        return

    train_df = st.session_state.train_data
    val_df = st.session_state.get("val_data")
    test_df = st.session_state.get("test_data")

    label_col = st.selectbox(
        "标签字段",
        train_df.columns,
        index=train_df.columns.get_loc(st.session_state.get("label_col", "label")),
    )
    st.session_state.label_col = label_col

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔍 特征筛选", "📦 特征衍生", "📊 特征分箱", "🔢 特征编码", "📈 稳定性分析"]
    )

    with tab1:
        st.subheader("特征筛选")
        filter_methods = st.multiselect(
            "筛选方法",
            ["缺失值过滤", "方差过滤", "IV值筛选", "相关性过滤"],
            default=["缺失值过滤", "方差过滤"],
        )

        col1, col2 = st.columns(2)
        with col1:
            missing_threshold = st.slider("缺失值阈值", 0.0, 1.0, 0.7)
            var_threshold = st.slider("方差阈值", 0.0, 1.0, 0.01)
        with col2:
            iv_threshold = st.slider("IV值阈值", 0.0, 1.0, 0.02)
            corr_threshold = st.slider("相关性阈值", 0.0, 1.0, 0.9)

        if st.button("执行筛选"):
            with st.spinner("正在筛选特征..."):
                selected_features = train_df.columns
                excluded_features = [label_col]

                # 缺失值过滤
                if "缺失值过滤" in filter_methods:
                    missing_rate = train_df.null_count() / len(train_df)
                    missing_filter = [
                        col
                        for col in selected_features
                        if missing_rate[col][0] < missing_threshold
                        and col not in excluded_features
                    ]
                    st.write(f"缺失值过滤：保留 {len(missing_filter)} 个特征")
                    selected_features = missing_filter

                # 方差过滤
                if "方差过滤" in filter_methods:
                    var = train_df.select(selected_features).var()
                    var_filter = [
                        col for col in selected_features if var[col][0] > var_threshold
                    ]
                    st.write(f"方差过滤：保留 {len(var_filter)} 个特征")
                    selected_features = var_filter

                # IV值筛选
                if "IV值筛选" in filter_methods:
                    iv_values = {}
                    for col in selected_features:
                        if train_df[col].dtype in [pl.Int64, pl.Float64]:
                            iv = calculate_iv(train_df, col, label_col)
                            iv_values[col] = iv
                    iv_filter = [
                        col for col, iv in iv_values.items() if iv >= iv_threshold
                    ]
                    st.write(f"IV值筛选：保留 {len(iv_filter)} 个特征")
                    selected_features = iv_filter

                    # 展示IV值
                    iv_df = pl.DataFrame(
                        {"feature": iv_values.keys(), "iv": iv_values.values()}
                    ).sort("iv", descending=True)
                    st.dataframe(iv_df, use_container_width=True)

                # 相关性过滤
                if "相关性过滤" in filter_methods:
                    corr = train_df.select(selected_features).corr()
                    corr_matrix = corr.to_numpy()
                    cols = corr.columns
                    to_drop = set()
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            if abs(corr_matrix[i][j]) > corr_threshold:
                                to_drop.add(cols[j])
                    corr_filter = [
                        col for col in selected_features if col not in to_drop
                    ]
                    st.write(f"相关性过滤：保留 {len(corr_filter)} 个特征")
                    selected_features = corr_filter

                st.session_state.selected_features = selected_features
                st.success(f"特征筛选完成！最终保留 {len(selected_features)} 个特征")
                st.write("选中特征：", selected_features)

    with tab2:
        st.subheader("特征衍生")
        numeric_features = [
            col
            for col in train_df.columns
            if train_df[col].dtype in [pl.Int64, pl.Float64] and col != label_col
        ]

        if "selected_features" in st.session_state:
            numeric_features = [
                col
                for col in numeric_features
                if col in st.session_state.selected_features
            ]

        derive_types = st.multiselect(
            "衍生类型",
            [
                "统计特征（均值/中位数/最大/最小）",
                "交叉特征（加减乘除）",
                "对数变换",
                "分位数特征",
            ],
        )

        if st.button("执行衍生"):
            with st.spinner("正在衍生特征..."):
                new_features = []
                derived_df = train_df.clone()

                if "统计特征" in str(derive_types):
                    for col in numeric_features:
                        derived_df = derived_df.with_columns(
                            [
                                pl.col(col).mean().over().alias(f"{col}_mean"),
                                pl.col(col).median().over().alias(f"{col}_median"),
                                pl.col(col).max().over().alias(f"{col}_max"),
                                pl.col(col).min().over().alias(f"{col}_min"),
                                pl.col(col).std().over().alias(f"{col}_std"),
                            ]
                        )
                        new_features.extend(
                            [
                                f"{col}_mean",
                                f"{col}_median",
                                f"{col}_max",
                                f"{col}_min",
                                f"{col}_std",
                            ]
                        )

                if "交叉特征" in str(derive_types):
                    for i in range(min(5, len(numeric_features))):
                        for j in range(i + 1, min(5, len(numeric_features))):
                            col1, col2 = numeric_features[i], numeric_features[j]
                            derived_df = derived_df.with_columns(
                                [
                                    (pl.col(col1) + pl.col(col2)).alias(
                                        f"{col1}_add_{col2}"
                                    ),
                                    (pl.col(col1) - pl.col(col2)).alias(
                                        f"{col1}_sub_{col2}"
                                    ),
                                    (pl.col(col1) * pl.col(col2)).alias(
                                        f"{col1}_mul_{col2}"
                                    ),
                                    (pl.col(col1) / (pl.col(col2) + 1e-10)).alias(
                                        f"{col1}_div_{col2}"
                                    ),
                                ]
                            )
                            new_features.extend(
                                [
                                    f"{col1}_add_{col2}",
                                    f"{col1}_sub_{col2}",
                                    f"{col1}_mul_{col2}",
                                    f"{col1}_div_{col2}",
                                ]
                            )

                if "对数变换" in str(derive_types):
                    for col in numeric_features:
                        derived_df = derived_df.with_columns(
                            pl.col(col).log1p().alias(f"{col}_log")
                        )
                        new_features.append(f"{col}_log")

                st.session_state.train_data = derived_df
                st.success(f"特征衍生完成！新增 {len(new_features)} 个特征")
                st.write(
                    "新增特征：",
                    new_features[:10],
                    "..." if len(new_features) > 10 else "",
                )

    with tab3:
        st.subheader("特征分箱")
        if "selected_features" not in st.session_state:
            st.warning("请先完成特征筛选！")
        else:
            feature_col = st.selectbox("选择特征", st.session_state.selected_features)
            bin_method = st.selectbox("分箱方法", ["等频分箱", "等宽分箱", "卡方分箱"])
            bin_num = st.slider("分箱数", 2, 20, 10)

            if st.button("执行分箱"):
                with st.spinner("正在分箱..."):
                    # 转pandas给toad处理
                    train_pd = train_df.select([feature_col, label_col]).to_pandas()

                    if bin_method == "等频分箱":
                        c = toad.transform.Combiner()
                        c.fit(train_pd, y=label_col, method="quantile", n_bins=bin_num)
                    elif bin_method == "等宽分箱":
                        c = toad.transform.Combiner()
                        c.fit(train_pd, y=label_col, method="step", n_bins=bin_num)
                    else:
                        c = toad.transform.Combiner()
                        c.fit(train_pd, y=label_col, method="chi", n_bins=bin_num)

                    # 分箱结果
                    bin_result = c.export()[feature_col]
                    st.write("分箱边界：", bin_result)

                    # 应用分箱
                    train_binned = c.transform(train_pd)
                    st.session_state.feature_binner = c

                    # 分箱统计
                    bin_stats = toad.BinStats().fit(
                        train_binned[feature_col], train_binned[label_col]
                    )
                    bin_stats_df = bin_stats.summary()
                    st.dataframe(bin_stats_df, use_container_width=True)

                    # 可视化
                    fig = px.bar(
                        bin_stats_df,
                        x="bin",
                        y=["bad_rate", "count"],
                        title=f"{feature_col} 分箱结果",
                        secondary_y="count",
                        labels={"value": "比例/数量", "bin": "分箱"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("特征编码")
        encode_method = st.selectbox(
            "编码方法", ["WOE编码", "独热编码", "标签编码", "标准化"]
        )

        if "feature_binner" not in st.session_state and encode_method == "WOE编码":
            st.warning("WOE编码需要先完成特征分箱！")
        elif st.button("执行编码"):
            with st.spinner("正在编码..."):
                if encode_method == "WOE编码":
                    woe_transformer = toad.transform.WOETransformer()
                    train_pd = train_df.to_pandas()
                    train_encoded = woe_transformer.fit_transform(
                        st.session_state.feature_binner.transform(train_pd),
                        train_pd[label_col],
                    )
                    st.session_state.train_data = pl.from_pandas(train_encoded)
                    st.session_state.woe_transformer = woe_transformer
                    st.success("WOE编码完成！")

                elif encode_method == "标准化":
                    selected_features = st.session_state.get(
                        "selected_features",
                        [col for col in train_df.columns if col != label_col],
                    )
                    train_encoded = train_df.clone()
                    for col in selected_features:
                        if train_encoded[col].dtype in [pl.Int64, pl.Float64]:
                            mean = train_encoded[col].mean()
                            std = train_encoded[col].std()
                            train_encoded = train_encoded.with_columns(
                                ((pl.col(col) - mean) / std).alias(col)
                            )
                    st.session_state.train_data = train_encoded
                    st.success("标准化完成！")

    with tab5:
        st.subheader("特征稳定性分析（PSI）")
        if val_df is None:
            st.warning("需要验证集才能计算PSI！")
        else:
            if "selected_features" not in st.session_state:
                st.warning("请先完成特征筛选！")
            else:
                feature_col = st.selectbox(
                    "选择特征", st.session_state.selected_features, key="psi_feature"
                )
                bin_num = st.slider("分箱数", 2, 20, 10, key="psi_bin")

                # 计算PSI
                psi, psi_df = calculate_psi(
                    train_df[feature_col],
                    val_df[feature_col],
                    bins=bin_num,
                    bin_type="equal_freq",
                )
                stability_level = get_feature_stability_level(psi)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("PSI值", f"{psi:.4f}")
                with col2:
                    st.metric("稳定性", stability_level)

                # PSI分布表
                st.dataframe(psi_df, use_container_width=True)

                # 可视化
                fig = px.bar(
                    psi_df,
                    x="break_point",
                    y=["expected_pct", "actual_pct"],
                    barmode="group",
                    title=f"{feature_col} 分布对比（训练集vs验证集）",
                    labels={"value": "占比", "break_point": "分箱边界"},
                )
                st.plotly_chart(fig, use_container_width=True)

                # 逐月PSI分析
                time_col = st.selectbox(
                    "选择时间字段（可选）",
                    [None]
                    + [
                        col
                        for col in train_df.columns
                        if train_df[col].dtype in [pl.Date, pl.Datetime]
                    ],
                )
                if time_col and st.button("计算逐月PSI"):
                    with st.spinner("正在计算逐月PSI..."):
                        # 按月份分组
                        train_with_month = train_df.with_columns(
                            pl.col(time_col).dt.strftime("%Y-%m").alias("month")
                        )
                        val_with_month = val_df.with_columns(
                            pl.col(time_col).dt.strftime("%Y-%m").alias("month")
                        )

                        months = sorted(val_with_month["month"].unique().to_list())
                        psi_trend = []

                        for month in months:
                            month_data = val_with_month.filter(pl.col("month") == month)
                            psi_month, _ = calculate_psi(
                                train_df[feature_col],
                                month_data[feature_col],
                                bins=bin_num,
                            )
                            psi_trend.append({"month": month, "psi": psi_month})

                        psi_trend_df = pl.DataFrame(psi_trend)
                        fig = px.line(
                            psi_trend_df,
                            x="month",
                            y="psi",
                            title=f"{feature_col} 逐月PSI趋势",
                            markers=True,
                        )
                        fig.add_hline(
                            y=0.1,
                            line_dash="dash",
                            line_color="green",
                            annotation_text="稳定阈值",
                        )
                        fig.add_hline(
                            y=0.25,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="漂移阈值",
                        )
                        st.plotly_chart(fig, use_container_width=True)
