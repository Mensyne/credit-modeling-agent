import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.metric_utils import calculate_psi


def render_visualization_agent():
    st.title("🎨 可视化Agent")

    tab1, tab2, tab3 = st.tabs(
        ["📊 特征分箱可视化", "📈 模型效果可视化", "📦 分数分箱可视化"]
    )

    with tab1:
        st.subheader("特征分箱可视化")
        if (
            "selected_features" not in st.session_state
            or "train_data" not in st.session_state
        ):
            st.warning("请先完成特征工程！")
        else:
            feature_col = st.selectbox("选择特征", st.session_state.selected_features)

            # 加载所有数据集
            datasets = {}
            if "train_data" in st.session_state:
                datasets["训练集"] = st.session_state.train_data
            if "val_data" in st.session_state:
                datasets["验证集"] = st.session_state.val_data
            if "test_data" in st.session_state:
                datasets["测试集"] = st.session_state.test_data

            selected_datasets = st.multiselect(
                "选择对比数据集", list(datasets.keys()), default=list(datasets.keys())
            )
            bin_num = st.slider("分箱数", 5, 20, 10)
            bin_type = st.selectbox("分箱方式", ["等频分箱", "等宽分箱"])

            if st.button("生成特征分箱图"):
                with st.spinner("正在生成图表..."):
                    # 计算分箱边界（使用训练集）
                    train_feat = datasets["训练集"][feature_col].to_numpy()
                    if bin_type == "等频分箱":
                        bins = np.percentile(
                            train_feat, np.linspace(0, 100, bin_num + 1)
                        )
                    else:
                        bins = np.linspace(
                            train_feat.min(), train_feat.max(), bin_num + 1
                        )

                    # 计算各数据集的分箱占比
                    bin_data = []
                    for ds_name in selected_datasets:
                        feat_data = datasets[ds_name][feature_col].to_numpy()
                        counts, _ = np.histogram(feat_data, bins=bins)
                        pct = counts / len(feat_data)

                        for i in range(len(bins) - 1):
                            bin_label = f"[{bins[i]:.4f}, {bins[i + 1]:.4f})"
                            bin_data.append(
                                {"分箱": bin_label, "占比": pct[i], "数据集": ds_name}
                            )

                    bin_df = pl.DataFrame(bin_data)
                    fig = px.bar(
                        bin_df,
                        x="分箱",
                        y="占比",
                        color="数据集",
                        barmode="group",
                        title=f"{feature_col} 不同数据集分箱对比",
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                    # 逐月分箱趋势
                    time_col = st.selectbox(
                        "选择时间字段（可选）",
                        [None]
                        + [
                            col
                            for col in datasets["训练集"].columns
                            if datasets["训练集"][col].dtype in [pl.Date, pl.Datetime]
                        ],
                    )
                    if time_col:
                        all_data = []
                        for ds_name in selected_datasets:
                            df = datasets[ds_name].with_columns(
                                pl.col(time_col).dt.strftime("%Y-%m").alias("month")
                            )
                            all_data.append(df)
                        combined_df = pl.concat(all_data)

                        months = sorted(combined_df["month"].unique().to_list())
                        month_bin_data = []

                        for month in months:
                            month_data = combined_df.filter(pl.col("month") == month)[
                                feature_col
                            ].to_numpy()
                            counts, _ = np.histogram(month_data, bins=bins)
                            pct = counts / len(month_data)

                            for i in range(len(bins) - 1):
                                bin_label = f"[{bins[i]:.4f}, {bins[i + 1]:.4f})"
                                month_bin_data.append(
                                    {"分箱": bin_label, "占比": pct[i], "月份": month}
                                )

                        month_bin_df = pl.DataFrame(month_bin_data)
                        fig = px.line(
                            month_bin_df,
                            x="月份",
                            y="占比",
                            color="分箱",
                            title=f"{feature_col} 逐月分箱趋势",
                            markers=True,
                        )
                        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("模型效果可视化")
        if "current_model" not in st.session_state:
            st.warning("请先训练模型！")
        else:
            model = st.session_state.current_model["model"]
            feature_cols = st.session_state.current_model["feature_cols"]
            label_col = st.session_state.current_model["label_col"]

            # 加载所有数据集
            datasets = {}
            if "train_data" in st.session_state:
                datasets["训练集"] = st.session_state.train_data
            if "val_data" in st.session_state:
                datasets["验证集"] = st.session_state.val_data
            if "test_data" in st.session_state:
                datasets["测试集"] = st.session_state.test_data

            selected_datasets = st.multiselect(
                "选择对比数据集", list(datasets.keys()), default=list(datasets.keys())
            )
            chart_type = st.selectbox(
                "图表类型", ["ROC曲线", "KS曲线", "PR曲线", "Lift曲线", "Gain曲线"]
            )

            if st.button("生成模型效果图"):
                with st.spinner("正在生成图表..."):
                    from sklearn.metrics import roc_curve, precision_recall_curve

                    fig = go.Figure()

                    for ds_name in selected_datasets:
                        df = datasets[ds_name]
                        X = df.select(feature_cols).to_pandas()
                        y_true = df[label_col].to_numpy()
                        y_pred = model.predict_proba(X)[:, 1]

                        if chart_type == "ROC曲线":
                            fpr, tpr, _ = roc_curve(y_true, y_pred)
                            auc = roc_auc_score(y_true, y_pred)
                            fig.add_trace(
                                go.Scatter(
                                    x=fpr,
                                    y=tpr,
                                    mode="lines",
                                    name=f"{ds_name} AUC={auc:.4f}",
                                )
                            )

                        elif chart_type == "KS曲线":
                            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                            ks = max(tpr - fpr)
                            fig.add_trace(
                                go.Scatter(
                                    x=thresholds,
                                    y=tpr - fpr,
                                    mode="lines",
                                    name=f"{ds_name} KS={ks:.4f}",
                                )
                            )

                        elif chart_type == "PR曲线":
                            precision, recall, _ = precision_recall_curve(
                                y_true, y_pred
                            )
                            ap = np.mean(precision)
                            fig.add_trace(
                                go.Scatter(
                                    x=recall,
                                    y=precision,
                                    mode="lines",
                                    name=f"{ds_name} AP={ap:.4f}",
                                )
                            )

                        elif chart_type in ["Lift曲线", "Gain曲线"]:
                            sorted_idx = np.argsort(y_pred)[::-1]
                            y_true_sorted = y_true[sorted_idx]
                            total_bad = y_true.sum()
                            total_samples = len(y_true)

                            percentages = np.arange(0.1, 1.1, 0.1)
                            values = []
                            for pct in percentages:
                                n = int(total_samples * pct)
                                bad_in_top = y_true_sorted[:n].sum()
                                gain = bad_in_top / total_bad
                                lift = gain / pct
                                values.append(
                                    lift if chart_type == "Lift曲线" else gain
                                )

                            fig.add_trace(
                                go.Scatter(
                                    x=percentages * 100,
                                    y=values,
                                    mode="lines+markers",
                                    name=ds_name,
                                )
                            )

                    # 辅助线
                    if chart_type == "ROC曲线":
                        fig.add_trace(
                            go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode="lines",
                                name="随机猜测",
                                line=dict(dash="dash"),
                            )
                        )
                        fig.update_layout(
                            title="ROC曲线对比",
                            xaxis_title="假阳性率(FPR)",
                            yaxis_title="真阳性率(TPR)",
                        )
                    elif chart_type == "KS曲线":
                        fig.update_layout(
                            title="KS曲线对比",
                            xaxis_title="阈值",
                            yaxis_title="KS值",
                            xaxis_autorange="reversed",
                        )
                    elif chart_type == "PR曲线":
                        fig.update_layout(
                            title="PR曲线对比",
                            xaxis_title="召回率",
                            yaxis_title="精确率",
                        )
                    elif chart_type == "Lift曲线":
                        fig.add_hline(
                            y=1,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="随机水平",
                        )
                        fig.update_layout(
                            title="Lift曲线对比",
                            xaxis_title="样本比例(%)",
                            yaxis_title="Lift值",
                        )
                    elif chart_type == "Gain曲线":
                        fig.add_trace(
                            go.Scatter(
                                x=[0, 100],
                                y=[0, 1],
                                mode="lines",
                                name="随机猜测",
                                line=dict(dash="dash"),
                            )
                        )
                        fig.update_layout(
                            title="Gain曲线对比",
                            xaxis_title="样本比例(%)",
                            yaxis_title="累计捕获率",
                        )

                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("模型分数分箱可视化")
        if "current_model" not in st.session_state:
            st.warning("请先训练模型！")
        else:
            model = st.session_state.current_model["model"]
            feature_cols = st.session_state.current_model["feature_cols"]
            label_col = st.session_state.current_model["label_col"]

            # 加载所有数据集
            datasets = {}
            if "train_data" in st.session_state:
                datasets["训练集"] = st.session_state.train_data
            if "val_data" in st.session_state:
                datasets["验证集"] = st.session_state.val_data
            if "test_data" in st.session_state:
                datasets["测试集"] = st.session_state.test_data

            selected_datasets = st.multiselect(
                "选择对比数据集", list(datasets.keys()), default=list(datasets.keys())
            )
            bin_num = st.slider("分箱数", 5, 20, 10)
            bin_type = st.selectbox("分箱方式", ["等频分箱", "等宽分箱"])

            if st.button("生成分数分箱图"):
                with st.spinner("正在生成图表..."):
                    # 计算分箱边界（使用训练集预测分数）
                    train_pred = model.predict_proba(
                        datasets["训练集"].select(feature_cols).to_pandas()
                    )[:, 1]
                    if bin_type == "等频分箱":
                        bins = np.percentile(
                            train_pred, np.linspace(0, 100, bin_num + 1)
                        )
                    else:
                        bins = np.linspace(
                            train_pred.min(), train_pred.max(), bin_num + 1
                        )

                    # 各数据集分数分箱对比
                    bin_data = []
                    bad_rate_data = []

                    for ds_name in selected_datasets:
                        df = datasets[ds_name]
                        y_pred = model.predict_proba(
                            df.select(feature_cols).to_pandas()
                        )[:, 1]
                        y_true = df[label_col].to_numpy()

                        counts, _ = np.histogram(y_pred, bins=bins)
                        pct = counts / len(y_pred)

                        # 计算坏样本率
                        bin_indices = np.digitize(y_pred, bins, right=True)
                        for i in range(1, bin_num + 1):
                            mask = bin_indices == i
                            if mask.sum() == 0:
                                continue
                            bin_label = (
                                f"[{bins[i - 1]:.4f}, {bins[i + 1]:.4f})"
                                if i < bin_num
                                else f"[{bins[i - 1]:.4f}, {bins[i]:.4f}]"
                            )
                            bad_rate = (
                                y_true[mask].mean() if len(y_true[mask]) > 0 else 0
                            )

                            bin_data.append(
                                {
                                    "分箱": bin_label,
                                    "样本占比": pct[i - 1],
                                    "数据集": ds_name,
                                }
                            )
                            bad_rate_data.append(
                                {
                                    "分箱": bin_label,
                                    "坏样本率": bad_rate,
                                    "数据集": ds_name,
                                }
                            )

                    # 样本分布对比
                    bin_df = pl.DataFrame(bin_data)
                    fig = px.bar(
                        bin_df,
                        x="分箱",
                        y="样本占比",
                        color="数据集",
                        barmode="group",
                        title="不同数据集分数分布对比",
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                    # 坏样本率对比
                    bad_rate_df = pl.DataFrame(bad_rate_data)
                    fig = go.Figure()
                    for ds_name in selected_datasets:
                        ds_data = bad_rate_df.filter(pl.col("数据集") == ds_name)
                        fig.add_trace(
                            go.Scatter(
                                x=ds_data["分箱"],
                                y=ds_data["坏样本率"],
                                mode="lines+markers",
                                name=ds_name,
                            )
                        )
                    fig.update_layout(
                        title="不同数据集坏样本率对比",
                        xaxis_title="分箱",
                        yaxis_title="坏样本率",
                        xaxis_tickangle=-45,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 逐月分数分箱趋势
                    time_col = st.selectbox(
                        "选择时间字段（可选）",
                        [None]
                        + [
                            col
                            for col in datasets["训练集"].columns
                            if datasets["训练集"][col].dtype in [pl.Date, pl.Datetime]
                        ],
                        key="score_time_col",
                    )
                    if time_col:
                        all_data = []
                        for ds_name in selected_datasets:
                            df = datasets[ds_name].with_columns(
                                pl.col(time_col).dt.strftime("%Y-%m").alias("month")
                            )
                            y_pred = model.predict_proba(
                                df.select(feature_cols).to_pandas()
                            )[:, 1]
                            df = df.with_columns(pl.Series(name="score", values=y_pred))
                            all_data.append(df)
                        combined_df = pl.concat(all_data)

                        months = sorted(combined_df["month"].unique().to_list())
                        month_data = []

                        for month in months:
                            month_df = combined_df.filter(pl.col("month") == month)
                            month_scores = month_df["score"].to_numpy()
                            counts, _ = np.histogram(month_scores, bins=bins)
                            pct = counts / len(month_scores)

                            for i in range(len(bins) - 1):
                                bin_label = f"[{bins[i]:.4f}, {bins[i + 1]:.4f})"
                                month_data.append(
                                    {"分箱": bin_label, "占比": pct[i], "月份": month}
                                )

                        month_df = pl.DataFrame(month_data)
                        fig = px.line(
                            month_df,
                            x="月份",
                            y="占比",
                            color="分箱",
                            title="模型分数逐月分布趋势",
                            markers=True,
                        )
                        st.plotly_chart(fig, use_container_width=True)


from sklearn.metrics import roc_auc_score
