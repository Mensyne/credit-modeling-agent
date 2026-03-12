import streamlit as st
import polars as pl
from sklearn.model_selection import train_test_split


def render_sample_split_agent():
    st.title("✂️ 样本划分Agent")

    if "raw_data" not in st.session_state:
        st.warning("请先上传/接入数据！")
        return

    df = st.session_state.raw_data
    st.write(f"当前数据集：{df.shape[0]} 行，{df.shape[1]} 列")

    split_type = st.radio("划分方式", ["随机划分", "时间划分"])
    label_col = st.selectbox("标签字段", df.columns)

    if split_type == "随机划分":
        train_size = st.slider("训练集比例", 0.5, 0.9, 0.7)
        val_size = st.slider("验证集比例", 0.1, 0.3, 0.2)
        test_size = 1 - train_size - val_size
        random_state = st.number_input("随机种子", value=42)

        if st.button("开始划分"):
            with st.spinner("正在划分样本..."):
                # 划分训练集和剩余集
                train, temp = train_test_split(
                    df.to_pandas(),
                    train_size=train_size,
                    random_state=random_state,
                    stratify=df[label_col].to_pandas(),
                )
                val, test = train_test_split(
                    temp,
                    train_size=val_size / (val_size + test_size),
                    random_state=random_state,
                    stratify=temp[label_col],
                )

                st.session_state.train_data = pl.from_pandas(train)
                st.session_state.val_data = pl.from_pandas(val)
                st.session_state.test_data = pl.from_pandas(test)

                st.success("样本划分完成！")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("训练集大小", f"{len(train)} 行")
                    st.metric("正样本占比", f"{train[label_col].mean():.2%}")
                with col2:
                    st.metric("验证集大小", f"{len(val)} 行")
                    st.metric("正样本占比", f"{val[label_col].mean():.2%}")
                with col3:
                    st.metric("测试集大小", f"{len(test)} 行")
                    st.metric("正样本占比", f"{test[label_col].mean():.2%}")

    else:  # 时间划分
        time_col = st.selectbox(
            "时间字段",
            [col for col in df.columns if df[col].dtype in [pl.Date, pl.Datetime]],
        )
        split_date = st.date_input("训练集截止日期")
        oot_split_date = st.date_input("OOT验证集截止日期（可选）", value=None)

        if st.button("开始划分"):
            with st.spinner("正在划分样本..."):
                train = df.filter(pl.col(time_col) <= pl.lit(split_date))
                if oot_split_date:
                    val = df.filter(
                        (pl.col(time_col) > split_date)
                        & (pl.col(time_col) <= pl.lit(oot_split_date))
                    )
                    test = df.filter(pl.col(time_col) > pl.lit(oot_split_date))
                else:
                    val = df.filter(pl.col(time_col) > pl.lit(split_date))
                    test = None

                st.session_state.train_data = train
                st.session_state.val_data = val
                if test is not None:
                    st.session_state.test_data = test

                st.success("样本划分完成！")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("训练集大小", f"{len(train)} 行")
                    st.metric("正样本占比", f"{train[label_col].mean():.2%}")
                with col2:
                    st.metric("验证集大小", f"{len(val)} 行")
                    st.metric("正样本占比", f"{val[label_col].mean():.2%}")
                with col3:
                    if test is not None:
                        st.metric("OOT测试集大小", f"{len(test)} 行")
                        st.metric("正样本占比", f"{test[label_col].mean():.2%}")
