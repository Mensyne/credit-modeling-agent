import streamlit as st
import polars as pl
import time
import psutil
import os
import plotly.express as px
import numpy as np


def render_performance_agent():
    st.title("⚡ 模型性能Agent")

    if "current_model" not in st.session_state:
        st.warning("请先训练模型！")
        return

    current_model = st.session_state.current_model
    model = current_model["model"]
    feature_cols = current_model["feature_cols"]

    tab1, tab2, tab3 = st.tabs(["⏱️ 推理性能", "💻 训练性能", "📊 多模型对比"])

    with tab1:
        st.subheader("推理性能测试")
        test_data_size = st.select_slider(
            "测试数据量", options=[100, 1000, 10000, 100000, 1000000], value=10000
        )

        if st.button("开始测试", type="primary"):
            with st.spinner("正在测试推理性能..."):
                # 生成测试数据
                X_test = np.random.rand(test_data_size, len(feature_cols))

                # 预热
                model.predict_proba(X_test[:100])

                # 测试单条推理
                single_start = time.time()
                model.predict_proba(X_test[:1])
                single_time = (time.time() - single_start) * 1000

                # 测试批量推理
                batch_start = time.time()
                model.predict_proba(X_test)
                batch_time = (time.time() - batch_start) * 1000
                qps = test_data_size / (batch_time / 1000)

                # 资源占用
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / 1024 / 1024
                cpu_usage = process.cpu_percent()

                # 展示结果
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("单条推理耗时", f"{single_time:.2f} ms")
                with col2:
                    st.metric(f"{test_data_size}条批量耗时", f"{batch_time:.2f} ms")
                with col3:
                    st.metric("吞吐量(QPS)", f"{qps:.0f} 条/秒")
                with col4:
                    st.metric("内存占用", f"{memory_usage:.2f} MB")

                st.metric("CPU使用率", f"{cpu_usage} %")

                # 不同数据量性能测试
                test_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
                if test_data_size > 100000:
                    test_sizes.append(test_data_size)

                performance_data = []
                for size in test_sizes:
                    if size > X_test.shape[0]:
                        continue
                    start = time.time()
                    model.predict_proba(X_test[:size])
                    elapsed = (time.time() - start) * 1000
                    performance_data.append(
                        {
                            "数据量": size,
                            "耗时(ms)": elapsed,
                            "QPS": size / (elapsed / 1000),
                        }
                    )

                perf_df = pl.DataFrame(performance_data)
                fig = px.line(
                    perf_df,
                    x="数据量",
                    y="耗时(ms)",
                    title="不同数据量推理耗时",
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)

                fig = px.line(
                    perf_df, x="数据量", y="QPS", title="不同数据量吞吐量", markers=True
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("训练性能统计")
        if "train_performance" not in current_model:
            st.info("暂无训练性能数据，下次训练时将自动记录")
        else:
            perf = current_model["train_performance"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("训练耗时", f"{perf['train_time']:.2f} 秒")
            with col2:
                st.metric("峰值内存占用", f"{perf['peak_memory']:.2f} MB")
            with col3:
                st.metric("平均CPU使用率", f"{perf['avg_cpu']} %")

            if "gpu_usage" in perf:
                st.metric("峰值GPU使用率", f"{perf['gpu_usage']} %")

    with tab3:
        st.subheader("多模型性能对比")
        if (
            "trained_models" not in st.session_state
            or len(st.session_state.trained_models) < 2
        ):
            st.warning("需要至少2个已训练模型才能对比")
        else:
            compare_data = []
            for m in st.session_state.trained_models:
                # 测试每个模型的推理性能
                X_test = np.random.rand(10000, len(m["feature_cols"]))
                start = time.time()
                m["model"].predict_proba(X_test)
                infer_time = (time.time() - start) * 1000

                compare_data.append(
                    {
                        "模型名称": m["name"],
                        "算法类型": m["type"],
                        "训练集AUC": m["train_auc"],
                        "验证集AUC": m["val_auc"] if m["val_auc"] else 0,
                        "1万条推理耗时(ms)": round(infer_time, 2),
                        "模型大小(MB)": round(
                            len(pickle.dumps(m["model"])) / 1024 / 1024, 2
                        ),
                    }
                )

            compare_df = pl.DataFrame(compare_data)
            st.dataframe(compare_df, use_container_width=True)

            # 可视化对比
            fig = px.bar(
                compare_df,
                x="模型名称",
                y=["训练集AUC", "验证集AUC"],
                barmode="group",
                title="各模型效果对比",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                compare_df,
                x="模型名称",
                y="1万条推理耗时(ms)",
                title="各模型推理性能对比",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                compare_df, x="模型名称", y="模型大小(MB)", title="各模型体积对比"
            )
            st.plotly_chart(fig, use_container_width=True)


import pickle
