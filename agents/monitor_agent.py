import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render_monitor_agent():
    st.title("📡 线上监控Agent")

    if "current_model" not in st.session_state:
        st.warning("请先选择模型！")
        return

    model = st.session_state.current_model

    # 监控时间范围
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期", value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input("结束日期", value=datetime.now())

    # 告警配置
    with st.expander("⚙️ 告警配置"):
        auc_threshold = st.slider("AUC告警阈值", 0.6, 0.9, value=0.7)
        ks_threshold = st.slider("KS告警阈值", 0.2, 0.5, value=0.3)
        psi_threshold = st.slider("PSI告警阈值", 0.1, 0.5, value=0.25)
        response_time_threshold = st.slider(
            "响应时间告警阈值(ms)", 100, 1000, value=500
        )
        alert_method = st.multiselect(
            "告警通知方式", ["邮件", "短信", "企业微信", "飞书"], default=["企业微信"]
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 效果监控", "⚡ 性能监控", "📊 特征漂移监控", "⚠️ 告警记录"]
    )

    # 生成模拟监控数据
    date_range = [
        start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
    ]
    n_days = len(date_range)

    with tab1:
        st.subheader("模型效果监控")
        # 模拟数据
        base_auc = model["val_auc"] if model["val_auc"] else 0.75
        base_ks = 0.4
        auc_data = [base_auc + np.random.normal(0, 0.02) for _ in range(n_days)]
        ks_data = [base_ks + np.random.normal(0, 0.03) for _ in range(n_days)]
        psi_data = [0.08 + abs(np.random.normal(0, 0.05)) for _ in range(n_days)]

        monitor_df = pl.DataFrame(
            {"日期": date_range, "AUC": auc_data, "KS": ks_data, "PSI": psi_data}
        )

        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            latest_auc = auc_data[-1]
            status = "✅ 正常" if latest_auc >= auc_threshold else "⚠️ 异常"
            st.metric("最新AUC", f"{latest_auc:.4f}", status)
        with col2:
            latest_ks = ks_data[-1]
            status = "✅ 正常" if latest_ks >= ks_threshold else "⚠️ 异常"
            st.metric("最新KS", f"{latest_ks:.4f}", status)
        with col3:
            latest_psi = psi_data[-1]
            status = "✅ 正常" if latest_psi < psi_threshold else "⚠️ 异常"
            st.metric("最新PSI", f"{latest_psi:.4f}", status)
        with col4:
            call_count = np.random.randint(1000, 10000)
            st.metric("累计调用量", f"{call_count} 次")

        # 趋势图
        fig = px.line(
            monitor_df, x="日期", y=["AUC", "KS"], title="AUC/KS趋势", markers=True
        )
        fig.add_hline(
            y=auc_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="AUC告警阈值",
        )
        fig.add_hline(
            y=ks_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text="KS告警阈值",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            monitor_df,
            x="日期",
            y="PSI",
            title="PSI趋势",
            markers=True,
            color_discrete_sequence=["red"],
        )
        fig.add_hline(
            y=psi_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="PSI告警阈值",
        )
        st.plotly_chart(fig, use_container_width=True)

        # 分群体效果
        st.subheader("分群体效果")
        group_data = pl.DataFrame(
            {
                "客群": [
                    "新客户",
                    "老客户",
                    "高收入",
                    "低收入",
                    "一线城市",
                    "二线及以下",
                ],
                "AUC": [0.78, 0.82, 0.85, 0.72, 0.81, 0.76],
                "样本量": [1200, 3500, 800, 3900, 1500, 3200],
            }
        )
        st.dataframe(group_data, use_container_width=True)

    with tab2:
        st.subheader("性能监控")
        # 模拟数据
        avg_response_time = [np.random.randint(100, 300) for _ in range(n_days)]
        p99_response_time = [np.random.randint(300, 800) for _ in range(n_days)]
        throughput = [np.random.randint(10, 100) for _ in range(n_days)]
        availability = [99.5 + np.random.normal(0, 0.3) for _ in range(n_days)]

        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            latest_avg_rt = avg_response_time[-1]
            status = "✅ 正常" if latest_avg_rt < response_time_threshold else "⚠️ 异常"
            st.metric("平均响应时间", f"{latest_avg_rt} ms", status)
        with col2:
            latest_p99_rt = p99_response_time[-1]
            st.metric("P99响应时间", f"{latest_p99_rt} ms")
        with col3:
            latest_tps = throughput[-1]
            st.metric("吞吐量", f"{latest_tps} QPS")
        with col4:
            latest_availability = availability[-1]
            st.metric("服务可用性", f"{latest_availability:.2f} %")

        # 趋势图
        perf_df = pl.DataFrame(
            {
                "日期": date_range,
                "平均响应时间(ms)": avg_response_time,
                "P99响应时间(ms)": p99_response_time,
                "吞吐量(QPS)": throughput,
                "可用性(%)": availability,
            }
        )

        fig = px.line(
            perf_df,
            x="日期",
            y=["平均响应时间(ms)", "P99响应时间(ms)"],
            title="响应时间趋势",
            markers=True,
        )
        fig.add_hline(
            y=response_time_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="响应时间告警阈值",
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                perf_df, x="日期", y="吞吐量(QPS)", title="吞吐量趋势", markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(
                perf_df, x="日期", y="可用性(%)", title="服务可用性趋势", markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("特征漂移监控")
        if "selected_features" in st.session_state:
            top_features = st.session_state.selected_features[:5]
            drift_data = []
            for feat in top_features:
                psi = 0.05 + abs(np.random.normal(0, 0.08))
                drift_data.append(
                    {
                        "特征名": feat,
                        "PSI值": round(psi, 4),
                        "漂移状态": "✅ 稳定"
                        if psi < 0.1
                        else "⚠️ 轻微漂移"
                        if psi < 0.25
                        else "❌ 严重漂移",
                        "变化趋势": "上升" if np.random.random() > 0.5 else "下降",
                    }
                )

            drift_df = pl.DataFrame(drift_data)
            st.dataframe(drift_df, use_container_width=True)

            # Top漂移特征
            st.subheader("Top 漂移特征趋势")
            selected_feat = st.selectbox("选择特征查看趋势", top_features)
            feat_psi = [0.06 + abs(np.random.normal(0, 0.03)) for _ in range(n_days)]
            feat_drift_df = pl.DataFrame({"日期": date_range, "PSI": feat_psi})
            fig = px.line(
                feat_drift_df,
                x="日期",
                y="PSI",
                title=f"{selected_feat} PSI趋势",
                markers=True,
            )
            fig.add_hline(
                y=0.1, line_dash="dash", line_color="green", annotation_text="稳定阈值"
            )
            fig.add_hline(
                y=0.25, line_dash="dash", line_color="red", annotation_text="漂移阈值"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("告警记录")
        # 模拟告警数据
        alerts = []
        for i in range(5):
            alert_date = end_date - timedelta(days=np.random.randint(0, 10))
            alert_type = np.random.choice(
                ["AUC低于阈值", "KS低于阈值", "PSI超过阈值", "响应时间超时"]
            )
            level = np.random.choice(["警告", "严重"])
            alerts.append(
                {
                    "告警时间": alert_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "告警类型": alert_type,
                    "告警级别": level,
                    "状态": np.random.choice(["已处理", "未处理"]),
                }
            )

        alerts_df = pl.DataFrame(alerts).sort("告警时间", descending=True)
        st.dataframe(alerts_df, use_container_width=True)

        # 告警统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("今日告警数", np.random.randint(0, 10))
        with col2:
            st.metric("未处理告警数", np.random.randint(0, 5))
        with col3:
            st.metric("告警处理率", f"{np.random.randint(80, 100)}%")
