import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
)
from scipy.stats import ks_2samp


def calculate_ks(y_true, y_pred):
    """计算KS值"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)
    return ks, fpr, tpr


def roc_curve(y_true, y_pred):
    """计算ROC曲线"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    desc_score_indices = np.argsort(y_pred, kind="mergesort")[::-1]
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_pred))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    return fpr, tpr, y_pred[threshold_idxs]


def calculate_psi(expected, actual, bins=10):
    """计算PSI"""
    expected_counts, bin_edges = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-10, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-10, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi, bin_edges, expected_pct, actual_pct


def render_effect_agent():
    st.title("📈 模型效果Agent")

    if "current_model" not in st.session_state:
        st.warning("请先训练模型！")
        return

    model = st.session_state.current_model["model"]
    feature_cols = st.session_state.current_model["feature_cols"]
    label_col = st.session_state.current_model["label_col"]

    # 加载数据集
    datasets = {}
    if "train_data" in st.session_state:
        datasets["训练集"] = st.session_state.train_data
    if "val_data" in st.session_state:
        datasets["验证集"] = st.session_state.val_data
    if "test_data" in st.session_state:
        datasets["测试集"] = st.session_state.test_data

    selected_dataset = st.selectbox("选择评估数据集", list(datasets.keys()))
    df = datasets[selected_dataset]

    # 预测
    X = df.select(feature_cols).to_pandas()
    y_true = df[label_col].to_numpy()
    y_pred = model.predict_proba(X)[:, 1]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 核心指标", "📈 曲线分析", "📦 分箱表现", "👥 分群体评估"]
    )

    with tab1:
        st.subheader("核心评估指标")
        # 计算指标
        auc = roc_auc_score(y_true, y_pred)
        ks, _, _ = calculate_ks(y_true, y_pred)
        psi = None
        if selected_dataset != "训练集":
            train_pred = model.predict_proba(
                datasets["训练集"].select(feature_cols).to_pandas()
            )[:, 1]
            psi, _, _, _ = calculate_psi(train_pred, y_pred)

        y_pred_label = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred_label)
        precision = precision_score(y_true, y_pred_label)
        recall = recall_score(y_true, y_pred_label)
        f1 = f1_score(y_true, y_pred_label)

        # 展示指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AUC", f"{auc:.4f}")
            st.metric("准确率", f"{accuracy:.4f}")
        with col2:
            st.metric("KS", f"{ks:.4f}")
            st.metric("精确率", f"{precision:.4f}")
        with col3:
            if psi is not None:
                st.metric("PSI", f"{psi:.4f}")
            else:
                st.metric("PSI", "-")
            st.metric("召回率", f"{recall:.4f}")
        with col4:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
            st.metric("F1值", f"{f1:.4f}")
            st.metric("混淆矩阵", f"TP:{tp} FP:{fp}<br>FN:{fn} TN:{tn}")

        # 混淆矩阵可视化
        cm = confusion_matrix(y_true, y_pred_label)
        fig = px.imshow(
            cm,
            labels=dict(x="预测标签", y="真实标签", color="数量"),
            x=["负样本", "正样本"],
            y=["负样本", "正样本"],
            title="混淆矩阵",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("曲线分析")
        curve_type = st.selectbox(
            "选择曲线类型", ["ROC曲线", "KS曲线", "PR曲线", "Lift曲线", "Gain曲线"]
        )

        if curve_type == "ROC曲线":
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.4f}")
            )
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
                title="ROC曲线",
                xaxis_title="假阳性率(FPR)",
                yaxis_title="真阳性率(TPR)",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif curve_type == "KS曲线":
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            ks_value = max(tpr - fpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=thresholds, y=tpr, mode="lines", name="TPR"))
            fig.add_trace(go.Scatter(x=thresholds, y=fpr, mode="lines", name="FPR"))
            fig.add_trace(
                go.Scatter(
                    x=thresholds, y=tpr - fpr, mode="lines", name=f"KS = {ks_value:.4f}"
                )
            )
            fig.update_layout(
                title="KS曲线",
                xaxis_title="阈值",
                yaxis_title="比例",
                xaxis_autorange="reversed",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif curve_type == "PR曲线":
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"AP = {np.mean(precision):.4f}",
                )
            )
            fig.update_layout(
                title="PR曲线", xaxis_title="召回率", yaxis_title="精确率"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif curve_type in ["Lift曲线", "Gain曲线"]:
            # 按预测分数降序排序
            sorted_idx = np.argsort(y_pred)[::-1]
            y_true_sorted = y_true[sorted_idx]
            total_bad = y_true.sum()
            total_samples = len(y_true)

            gains = []
            lifts = []
            percentages = np.arange(0.1, 1.1, 0.1)
            for pct in percentages:
                n = int(total_samples * pct)
                bad_in_top = y_true_sorted[:n].sum()
                gain = bad_in_top / total_bad
                lift = gain / pct
                gains.append(gain)
                lifts.append(lift)

            if curve_type == "Lift曲线":
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=percentages * 100,
                        y=lifts,
                        mode="lines+markers",
                        name="Lift值",
                    )
                )
                fig.add_hline(
                    y=1, line_dash="dash", line_color="red", annotation_text="随机水平"
                )
                fig.update_layout(
                    title="Lift曲线", xaxis_title="样本比例(%)", yaxis_title="Lift值"
                )
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=percentages * 100,
                        y=np.array(gains) * 100,
                        mode="lines+markers",
                        name="累计捕获率",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 100],
                        y=[0, 100],
                        mode="lines",
                        name="随机猜测",
                        line=dict(dash="dash"),
                    )
                )
                fig.update_layout(
                    title="Gain曲线",
                    xaxis_title="样本比例(%)",
                    yaxis_title="累计坏客户捕获率(%)",
                )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("分数分箱表现")
        bin_num = st.slider("分箱数", 5, 20, 10)
        bin_type = st.selectbox("分箱方式", ["等频分箱", "等宽分箱"])

        # 分箱
        if bin_type == "等频分箱":
            bins = np.percentile(y_pred, np.linspace(0, 100, bin_num + 1))
        else:
            bins = np.linspace(y_pred.min(), y_pred.max(), bin_num + 1)

        bin_indices = np.digitize(y_pred, bins, right=True)
        bin_stats = []

        for i in range(1, bin_num + 1):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            bin_min = round(bins[i - 1], 4)
            bin_max = round(bins[i], 4)
            total = mask.sum()
            bad = y_true[mask].sum()
            bad_rate = bad / total
            bin_stats.append(
                {
                    "分箱区间": f"[{bin_min}, {bin_max})",
                    "样本数": total,
                    "坏样本数": bad,
                    "坏样本率": f"{bad_rate:.2%}",
                    "占总样本比例": f"{total / len(y_true):.2%}",
                    "占总坏样本比例": f"{bad / y_true.sum():.2%}",
                }
            )

        st.dataframe(pl.DataFrame(bin_stats), use_container_width=True)

        # 可视化
        bin_stats_df = pl.DataFrame(bin_stats)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=bin_stats_df["分箱区间"],
                y=bin_stats_df["样本数"],
                name="样本数",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bin_stats_df["分箱区间"],
                y=[float(x.strip("%")) / 100 for x in bin_stats_df["坏样本率"]],
                name="坏样本率",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="red"),
            )
        )
        fig.update_layout(
            title="分箱表现",
            yaxis=dict(title="样本数"),
            yaxis2=dict(
                title="坏样本率", overlaying="y", side="right", tickformat=".1%"
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("分群体效果评估")
        group_cols = st.multiselect(
            "选择分组字段",
            [col for col in df.columns if col not in feature_cols + [label_col]],
        )

        if group_cols:
            group_col = st.selectbox("按字段分组", group_cols)
            # 计算各群体指标
            group_results = []
            groups = df[group_col].unique().to_list()

            for group in groups:
                group_mask = df[group_col] == group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]

                if len(group_y_true) < 100:
                    continue

                group_auc = roc_auc_score(group_y_true, group_y_pred)
                group_ks, _, _ = calculate_ks(group_y_true, group_y_pred)
                group_bad_rate = group_y_true.mean()

                group_results.append(
                    {
                        "群体": group,
                        "样本数": len(group_y_true),
                        "坏样本率": f"{group_bad_rate:.2%}",
                        "AUC": f"{group_auc:.4f}",
                        "KS": f"{group_ks:.4f}",
                    }
                )

            st.dataframe(pl.DataFrame(group_results), use_container_width=True)

            # 可视化对比
            result_df = pl.DataFrame(group_results)
            fig = px.bar(
                result_df,
                x="群体",
                y=[float(x.strip("%")) / 100 for x in result_df["坏样本率"]],
                title="各群体坏样本率对比",
                labels={"y": "坏样本率"},
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                result_df,
                x="群体",
                y=[float(x) for x in result_df["AUC"]],
                title="各群体AUC对比",
                labels={"y": "AUC"},
            )
            st.plotly_chart(fig, use_container_width=True)
