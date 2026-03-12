import polars as pl
import pandas as pd
import json
import requests
from sqlalchemy import create_engine
from typing import Optional, Union


def load_file(file_obj) -> Optional[pl.DataFrame]:
    """加载本地文件"""
    try:
        file_type = file_obj.name.split(".")[-1].lower()
        if file_type == "csv":
            return pl.read_csv(file_obj)
        elif file_type in ["xlsx", "xls"]:
            return pl.from_pandas(pd.read_excel(file_obj))
        elif file_type == "parquet":
            return pl.read_parquet(file_obj)
        else:
            raise ValueError(f"不支持的文件类型：{file_type}")
    except Exception as e:
        import streamlit as st

        st.error(f"文件加载失败：{str(e)}")
        return None


def connect_database(
    db_type: str,
    host: str,
    port: int,
    user: str,
    password: str,
    db_name: str,
    query: str,
) -> Optional[pl.DataFrame]:
    """连接数据库并查询数据"""
    try:
        if db_type == "MySQL":
            conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
        elif db_type == "ClickHouse":
            conn_str = f"clickhouse+native://{user}:{password}@{host}:{port}/{db_name}"
        elif db_type == "PostgreSQL":
            conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        else:
            raise ValueError(f"不支持的数据库类型：{db_type}")

        engine = create_engine(conn_str)
        df = pd.read_sql(query, engine)
        return pl.from_pandas(df)
    except Exception as e:
        import streamlit as st

        st.error(f"数据库查询失败：{str(e)}")
        return None


def fetch_api_data(
    url: str,
    method: str = "GET",
    headers: Optional[str] = None,
    params: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    """从API获取数据"""
    try:
        headers_dict = json.loads(headers) if headers else {}
        params_dict = json.loads(params) if params else {}

        if method == "GET":
            resp = requests.get(
                url, headers=headers_dict, params=params_dict, timeout=30
            )
        elif method == "POST":
            resp = requests.post(
                url, headers=headers_dict, json=params_dict, timeout=30
            )
        else:
            raise ValueError(f"不支持的请求方法：{method}")

        resp.raise_for_status()
        data = resp.json()
        return pl.DataFrame(data)
    except Exception as e:
        import streamlit as st

        st.error(f"API请求失败：{str(e)}")
        return None
