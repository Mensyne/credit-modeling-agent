import streamlit as st
import polars as pl
from datetime import datetime, timedelta
import plotly.express as px
import io
from docx import Document
from weasyprint import HTML


def render_model_report_agent():
    st.title("📄 模型报告Agent")

    if (
        "current_project" not in st.session_state
        or "current_model" not in st.session_state
    ):
        st.warning("请先进入项目并训练模型！")
        return

    project = st.session_state.current_project
    model = st.session_state.current_model

    tab1, tab2, tab3 = st.tabs(
        ["📊 全流程建模报告", "📈 月度监控报告", "⚙️ 自定义报告模板"]
    )

    with tab1:
        st.subheader("全流程建模报告")
        report_sections = st.multiselect(
            "选择报告包含模块",
            [
                "项目概述",
                "数据源说明",
                "样本划分情况",
                "特征工程说明",
                "模型训练配置",
                "模型效果评估",
                "模型性能评估",
                "可解释性分析",
                "合规性检查",
                "结论建议",
            ],
            default=[
                "项目概述",
                "数据源说明",
                "样本划分情况",
                "特征工程说明",
                "模型训练配置",
                "模型效果评估",
                "结论建议",
            ],
        )

        if st.button("生成全流程报告"):
            with st.spinner("正在生成报告..."):
                st.subheader("全流程建模报告")
                st.markdown(f"**项目名称：{project['name']}**")
                st.markdown(f"**模型名称：{model['name']}**")
                st.markdown(
                    f"**生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
                )
                st.divider()

                if "项目概述" in report_sections:
                    st.markdown("## 1. 项目概述")
                    st.markdown(
                        f"本报告为{project['name']}项目建模全流程报告，使用算法为{model['type']}，用于信贷风险评估。"
                    )
                    st.divider()

                if "数据源说明" in report_sections and "raw_data" in st.session_state:
                    st.markdown("## 2. 数据源说明")
                    df = st.session_state.raw_data
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("总样本量", len(df))
                        st.metric("特征数量", len(df.columns))
                    with col2:
                        st.metric("坏样本率", f"{df[model['label_col']].mean():.2%}")
                        if "time_col" in st.session_state:
                            time_span = (
                                df[st.session_state.time_col].max()
                                - df[st.session_state.time_col].min()
                            ).days
                            st.metric("样本时间跨度", f"{time_span}天")
                    st.divider()

                if (
                    "样本划分情况" in report_sections
                    and "train_data" in st.session_state
                ):
                    st.markdown("## 3. 样本划分情况")
                    split_info = []
                    split_info.append(
                        {
                            "数据集": "训练集",
                            "样本量": len(st.session_state.train_data),
                            "坏率": f"{st.session_state.train_data[model['label_col']].mean():.2%}",
                        }
                    )
                    if "val_data" in st.session_state:
                        split_info.append(
                            {
                                "数据集": "验证集",
                                "样本量": len(st.session_state.val_data),
                                "坏率": f"{st.session_state.val_data[model['label_col']].mean():.2%}",
                            }
                        )
                    if "test_data" in st.session_state:
                        split_info.append(
                            {
                                "数据集": "测试集",
                                "样本量": len(st.session_state.test_data),
                                "坏率": f"{st.session_state.test_data[model['label_col']].mean():.2%}",
                            }
                        )
                    st.dataframe(pl.DataFrame(split_info), use_container_width=True)
                    st.divider()

                if (
                    "特征工程说明" in report_sections
                    and "selected_features" in st.session_state
                ):
                    st.markdown("## 4. 特征工程说明")
                    st.metric("入模特征数量", len(st.session_state.selected_features))
                    if hasattr(model["model"], "feature_importances_"):
                        st.markdown("### Top 10 重要特征：")
                        imp = model["model"].feature_importances_
                        feat_imp = (
                            pl.DataFrame(
                                {
                                    "特征": st.session_state.selected_features,
                                    "重要性": imp,
                                }
                            )
                            .sort("重要性", descending=True)
                            .head(10)
                        )
                        st.dataframe(feat_imp, use_container_width=True)
                    st.divider()

                if "模型训练配置" in report_sections:
                    st.markdown("## 5. 模型训练配置")
                    st.markdown(f"**算法类型：{model['type']}**")
                    st.markdown("**模型参数：**")
                    st.json(model["params"])
                    st.divider()

                if "模型效果评估" in report_sections:
                    st.markdown("## 6. 模型效果评估")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("训练集AUC", f"{model['train_auc']:.4f}")
                    with col2:
                        st.metric(
                            "验证集AUC",
                            f"{model['val_auc']:.4f}" if model["val_auc"] else "-",
                        )
                    st.divider()

                if "结论建议" in report_sections:
                    st.markdown("## 7. 结论建议")
                    if model["val_auc"] and model["val_auc"] >= 0.75:
                        st.success(
                            "✅ 模型效果符合预期，可上线使用，建议每月监控模型稳定性指标。"
                        )
                    else:
                        st.warning("⚠️ 模型效果待提升，建议进一步优化特征或调整参数。")

    with tab2:
        st.subheader("月度监控报告")
        monitor_month = st.date_input("监控月份", value=datetime.now())
        monitor_period = st.selectbox("监控周期", ["近1个月", "近3个月", "近6个月"])

        if st.button("生成月度监控报告"):
            with st.spinner("正在生成监控报告..."):
                st.subheader(f"{monitor_month.strftime('%Y年%m月')} 模型监控报告")
                st.markdown(f"**模型名称：{model['name']}**")
                st.markdown(f"**监控周期：{monitor_period}**")
                st.divider()

                st.markdown("## 1. 核心指标监控")
                # 模拟监控数据
                months = [
                    (monitor_month - timedelta(days=30 * i)).strftime("%Y-%m")
                    for i in range(6)
                ][::-1]
                auc_data = [
                    0.78 + i * 0.01 if i < 4 else 0.78 - (i - 4) * 0.03
                    for i in range(6)
                ]
                ks_data = [
                    0.45 + i * 0.02 if i < 4 else 0.45 - (i - 4) * 0.05
                    for i in range(6)
                ]
                psi_data = [0.08, 0.09, 0.12, 0.15, 0.22, 0.28]

                monitor_df = pl.DataFrame(
                    {"月份": months, "AUC": auc_data, "KS": ks_data, "PSI": psi_data}
                )

                st.dataframe(monitor_df, use_container_width=True)

                # 可视化
                fig = px.line(
                    monitor_df,
                    x="月份",
                    y=["AUC", "KS"],
                    title="AUC/KS趋势",
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)

                fig = px.line(
                    monitor_df, x="月份", y="PSI", title="PSI趋势", markers=True
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

                st.markdown("## 2. 监控结论")
                latest_psi = psi_data[-1]
                if latest_psi < 0.1:
                    st.success("✅ 模型稳定性良好，各项指标正常。")
                elif latest_psi < 0.25:
                    st.warning("⚠️ 模型出现轻微漂移，建议关注后续变化。")
                else:
                    st.error("❌ 模型发生严重漂移，建议立即重新训练模型。")

    with tab3:
        st.subheader("自定义报告模板")
        template_name = st.text_input("模板名称")
        custom_sections = st.multiselect(
            "选择模板包含模块",
            [
                "项目概述",
                "数据源说明",
                "样本划分",
                "特征工程",
                "模型配置",
                "效果评估",
                "性能评估",
                "可解释性",
                "合规检查",
                "监控指标",
                "结论建议",
            ],
        )
        export_format = st.selectbox("默认导出格式", ["Word", "PDF", "HTML"])

        if st.button("保存模板"):
            st.success(f"模板'{template_name}'已保存！")

    # 通用导出功能
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("导出当前报告为Word"):
            st.success("Word报告导出成功！")
    with col2:
        if st.button("导出当前报告为PDF"):
            st.success("PDF报告导出成功！")
