import streamlit as st
import polars as pl
import os
from utils.io_utils import load_file, connect_database, fetch_api_data


def render_data_agent():
    st.title("📥 数据Agent")

    if "current_project" not in st.session_state:
        st.warning("请先进入项目！")
        return

    project = st.session_state.current_project
    st.write(f"当前项目：{project['name']}")

    tab1, tab2, tab3 = st.tabs(["📁 本地上传", "🗄️ 数据库对接", "🔌 API接入"])

    with tab1:
        st.subheader("本地文件上传")
        uploaded_file = st.file_uploader("选择文件", type=["csv", "xlsx", "parquet"])
        if uploaded_file:
            with st.spinner("正在加载数据..."):
                try:
                    df = load_file(uploaded_file)
                    st.session_state.raw_data = df
                    st.success(f"数据加载成功！共 {df.shape[0]} 行，{df.shape[1]} 列")
                    st.subheader("数据预览")
                    st.dataframe(df.head(10))

                    # 数据探查
                    st.subheader("数据探查")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("字段统计")
                        st.dataframe(df.describe())
                    with col2:
                        st.write("缺失值统计")
                        missing = df.null_count()
                        st.dataframe(missing)
                except Exception as e:
                    st.error(f"文件加载失败：{str(e)}")

    with tab2:
        st.subheader("数据库对接")
        db_type = st.selectbox("数据库类型", ["MySQL", "ClickHouse", "PostgreSQL"])
        host = st.text_input("主机地址")
        port = st.number_input(
            "端口",
            value=3306
            if db_type == "MySQL"
            else 8123
            if db_type == "ClickHouse"
            else 5432,
        )
        user = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        db_name = st.text_input("数据库名")
        query = st.text_area("查询SQL")

        if st.button("连接并查询"):
            with st.spinner("正在连接数据库..."):
                try:
                    df = connect_database(
                        db_type, host, port, user, password, db_name, query
                    )
                    st.session_state.raw_data = df
                    st.success(f"数据查询成功！共 {df.shape[0]} 行，{df.shape[1]} 列")
                    st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"数据库查询失败：{str(e)}")

    with tab3:
        st.subheader("API接入")
        api_url = st.text_input("API地址")
        method = st.selectbox("请求方法", ["GET", "POST"])
        headers = st.text_area("请求头（JSON格式）")
        params = st.text_area("请求参数（JSON格式）")

        if st.button("请求数据"):
            with st.spinner("正在请求API..."):
                try:
                    df = fetch_api_data(api_url, method, headers, params)
                    st.session_state.raw_data = df
                    st.success(f"数据获取成功！共 {df.shape[0]} 行，{df.shape[1]} 列")
                    st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"API请求失败：{str(e)}")
