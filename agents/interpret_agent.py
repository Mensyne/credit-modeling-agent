import streamlit as st
import polars as pl
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import PartialDependenceDisplay


def render_interpret_agent():
    st.title("🔍 可解释性Agent")

    if "current_model" not in st.session_state or "train_data" not in st.session_state:
        st.warning("请先训练模型！")
        return

    model = st.session_state.current_model["model"]
    feature_cols = st.session_state.current_model["feature_cols"]
    label_col = st.session_state.current_model["label_col"]
    train_df = st.session_state.train_data

    X_train = train_df.select(feature_cols).to_pandas()

    # 业务特征映射配置
    with st.expander("⚙️ 业务特征映射配置（可选）"):
        st.info("可将特征英文名称映射为业务含义，提升可解释性可读性")
        feature_mapping = {}
        for col in feature_cols:
            feature_mapping[col] = st.text_input(f"{col} 业务含义", value=col)

    tab1, tab2, tab3 = st.tabs(["🌐 全局解释", "🔍 局部解释", "📊 部分依赖图"])

    with tab1:
        st.subheader("全局模型解释")
        explain_type = st.selectbox("解释类型", ["特征重要性", "SHAP值分析"])

        if explain_type == "特征重要性":
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feat_imp = (
                    pl.DataFrame(
                        {
                            "feature": [
                                feature_mapping.get(col, col) for col in feature_cols
                            ],
                            "importance": importances,
                        }
                    )
                    .sort("importance", descending=True)
                    .head(20)
                )

                fig = px.bar(
                    feat_imp,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 20 特征重要性",
                    labels={"importance": "重要性得分", "feature": "特征"},
                )
                st.plotly_chart(fig, use_container_width=True)

                # 展示完整重要性表
                st.subheader("完整特征重要性")
                full_feat_imp = pl.DataFrame(
                    {
                        "特征": [feature_mapping.get(col, col) for col in feature_cols],
                        "重要性得分": importances,
                        "原始特征名": feature_cols,
                    }
                ).sort("重要性得分", descending=True)
                st.dataframe(full_feat_imp, use_container_width=True)
            else:
                st.info("当前模型不支持原生特征重要性计算，请使用SHAP值分析")

        elif explain_type == "SHAP值分析":
            with st.spinner("正在计算SHAP值..."):
                # 初始化SHAP解释器
                try:
                    if hasattr(model, "predict_proba"):
                        explainer = (
                            shap.TreeExplainer(model)
                            if hasattr(model, "get_booster")
                            or "LGBM" in str(type(model))
                            or "CatBoost" in str(type(model))
                            else shap.KernelExplainer(
                                model.predict_proba, shap.sample(X_train, 100)
                            )
                        )
                        shap_values = explainer.shap_values(X_train)

                        # 二分类取正例的SHAP值
                        if isinstance(shap_values, list) and len(shap_values) == 2:
                            shap_values = shap_values[1]

                        st.subheader("SHAP值汇总图")
                        fig, ax = plt.subplots()
                        shap.summary_plot(
                            shap_values,
                            X_train,
                            feature_names=[
                                feature_mapping.get(col, col) for col in feature_cols
                            ],
                            show=False,
                        )
                        st.pyplot(fig)

                        st.subheader("SHAP蜂群图")
                        fig, ax = plt.subplots()
                        shap.plots.beeswarm(
                            shap.Explanation(
                                values=shap_values,
                                data=X_train,
                                feature_names=[
                                    feature_mapping.get(col, col)
                                    for col in feature_cols
                                ],
                            ),
                            show=False,
                        )
                        st.pyplot(fig)

                        # SHAP依赖图
                        st.subheader("SHAP依赖图")
                        selected_feature = st.selectbox(
                            "选择特征",
                            [feature_mapping.get(col, col) for col in feature_cols],
                        )
                        original_feat = [
                            k
                            for k, v in feature_mapping.items()
                            if v == selected_feature
                        ][0]

                        fig, ax = plt.subplots()
                        shap.dependence_plot(
                            original_feat,
                            shap_values,
                            X_train,
                            feature_names=[
                                feature_mapping.get(col, col) for col in feature_cols
                            ],
                            show=False,
                        )
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"SHAP计算失败：{str(e)}")
                    st.info(
                        "当前模型暂不支持SHAP分析，可尝试使用树模型（XGBoost/LightGBM/CatBoost）"
                    )

    with tab2:
        st.subheader("单样本局部解释")
        # 选择样本
        sample_idx = st.number_input(
            "选择样本索引", min_value=0, max_value=len(train_df) - 1, value=0
        )
        sample = X_train.iloc[[sample_idx]]
        sample_original = train_df[sample_idx].to_dict(as_series=False)

        # 预测结果
        pred_proba = model.predict_proba(sample)[0][1]
        true_label = sample_original[label_col][0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("预测违约概率", f"{pred_proba:.2%}")
        with col2:
            st.metric("真实标签", "违约" if true_label == 1 else "未违约")
        with col3:
            st.metric("预测结果", "高风险" if pred_proba >= 0.5 else "低风险")

        # 展示样本详情
        with st.expander("查看样本详情"):
            sample_df = pl.DataFrame(
                {
                    "特征名": [feature_mapping.get(col, col) for col in feature_cols],
                    "特征值": [sample_original[col][0] for col in feature_cols],
                }
            )
            st.dataframe(sample_df, use_container_width=True)

        # SHAP局部解释
        if st.button("计算特征贡献度"):
            with st.spinner("正在计算..."):
                try:
                    explainer = (
                        shap.TreeExplainer(model)
                        if hasattr(model, "get_booster")
                        or "LGBM" in str(type(model))
                        or "CatBoost" in str(type(model))
                        else shap.KernelExplainer(
                            model.predict_proba, shap.sample(X_train, 100)
                        )
                    )
                    shap_value = explainer.shap_values(sample)

                    if isinstance(shap_value, list) and len(shap_value) == 2:
                        shap_value = shap_value[1][0]
                    else:
                        shap_value = shap_value[0]

                    # 计算特征贡献
                    contribution = []
                    for i, col in enumerate(feature_cols):
                        contribution.append(
                            {
                                "特征": feature_mapping.get(col, col),
                                "特征值": sample_original[col][0],
                                "贡献度": shap_value[i],
                                "影响方向": "正向（提升违约概率）"
                                if shap_value[i] > 0
                                else "负向（降低违约概率）",
                            }
                        )

                    contribution_df = pl.DataFrame(contribution).sort(
                        "贡献度", descending=True, key=abs
                    )

                    # 展示贡献度排名
                    st.subheader("特征贡献度排名")
                    st.dataframe(contribution_df, use_container_width=True)

                    # 可视化
                    top_n = st.slider("展示Top N特征", 5, 20, 10)
                    top_contribution = contribution_df.head(top_n)

                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=top_contribution["贡献度"],
                            y=top_contribution["特征"],
                            orientation="h",
                            marker_color=[
                                "red" if x > 0 else "green"
                                for x in top_contribution["贡献度"]
                            ],
                        )
                    )
                    fig.update_layout(
                        title=f"Top {top_n} 特征对预测结果的贡献",
                        xaxis_title="贡献度（正：提升违约概率，负：降低违约概率）",
                        yaxis_title="特征",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"计算失败：{str(e)}")

    with tab3:
        st.subheader("部分依赖图（PDP）")
        st.info("展示特征取值变化对模型预测结果的边际影响")

        selected_features = st.multiselect(
            "选择特征（最多2个）",
            feature_cols,
            default=[feature_cols[0]] if feature_cols else [],
        )

        if len(selected_features) > 0 and st.button("生成部分依赖图"):
            with st.spinner("正在生成..."):
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    PartialDependenceDisplay.from_estimator(
                        model,
                        X_train,
                        features=selected_features,
                        feature_names=[
                            feature_mapping.get(col, col) for col in feature_cols
                        ],
                        ax=ax,
                    )
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"生成失败：{str(e)}")


import matplotlib.pyplot as plt

plt.style.use("default")
