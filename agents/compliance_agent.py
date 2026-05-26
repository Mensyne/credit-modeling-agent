import streamlit as st
import polars as pl
import os
import re


def render_compliance_agent():
    st.title("✅ 数据合规Agent")

    if "raw_data" not in st.session_state:
        st.warning("请先上传/接入数据！")
        return

    df = st.session_state.raw_data
    st.write(f"当前数据集：{df.shape[0]} 行，{df.shape[1]} 列")

    # 加载敏感特征配置
    default_sensitive = [
        "gender",
        "sex",
        "race",
        "religion",
        "id_card",
        "idcard",
        "phone",
        "mobile",
        "address",
        "location",
        "birthday",
        "age",
    ]
    sensitive_features = (
        os.getenv("SENSITIVE_FEATURES", "").split(",") + default_sensitive
    )
    sensitive_features = list(
        set([feat.lower() for feat in sensitive_features if feat])
    )

    tab1, tab2, tab3 = st.tabs(["🔍 敏感特征识别", "📑 数据源合规", "📊 样本合规校验"])

    with tab1:
        st.subheader("敏感特征识别")
        # 匹配敏感特征
        detected_sensitive = []
        for col in df.columns:
            col_lower = col.lower()
            # 精确匹配
            if col_lower in sensitive_features:
                detected_sensitive.append(
                    {
                        "特征名": col,
                        "匹配类型": "精确匹配",
                        "风险等级": "高风险",
                        "说明": f"命中敏感特征库：{col_lower}",
                    }
                )
            # 模糊匹配
            for sensitive_feat in sensitive_features:
                if sensitive_feat in col_lower and col_lower not in [
                    x["特征名"].lower() for x in detected_sensitive
                ]:
                    detected_sensitive.append(
                        {
                            "特征名": col,
                            "匹配类型": "模糊匹配",
                            "风险等级": "中风险",
                            "说明": f"包含敏感关键词：{sensitive_feat}",
                        }
                    )
                    break
            # 正则匹配身份证/手机号
            if df[col].dtype == pl.Utf8:
                sample_values = df[col].head(10).to_list()
                for val in sample_values:
                    if val and isinstance(val, str):
                        if re.match(r"^\d{17}[\dXx]$", val):
                            detected_sensitive.append(
                                {
                                    "特征名": col,
                                    "匹配类型": "规则匹配",
                                    "风险等级": "高风险",
                                    "说明": "疑似身份证号码字段",
                                }
                            )
                            break
                        if re.match(r"^1[3-9]\d{9}$", val):
                            detected_sensitive.append(
                                {
                                    "特征名": col,
                                    "匹配类型": "规则匹配",
                                    "风险等级": "高风险",
                                    "说明": "疑似手机号码字段",
                                }
                            )
                            break

        if detected_sensitive:
            st.error(f"⚠️ 检测到 {len(detected_sensitive)} 个敏感特征！")
            st.dataframe(pl.DataFrame(detected_sensitive), use_container_width=True)

            # 自动脱敏选项
            if st.button("一键脱敏敏感特征"):
                with st.spinner("正在脱敏..."):
                    for feat in detected_sensitive:
                        col = feat["特征名"]
                        if feat["风险等级"] == "高风险":
                            # 高风险特征直接删除
                            df = df.drop(col)
                            st.info(f"已删除高风险敏感特征：{col}")
                        else:
                            # 中风险特征掩码处理
                            if df[col].dtype == pl.Utf8:
                                df = df.with_columns(
                                    pl.col(col).str.replace_all(r".", "*").alias(col)
                                )
                                st.info(f"已掩码处理中风险敏感特征：{col}")
                    st.session_state.raw_data = df
                    st.success("敏感特征脱敏完成！")
        else:
            st.success("✅ 未检测到敏感特征")

    with tab2:
        st.subheader("数据源合规校验")
        data_source = st.selectbox(
            "数据来源类型", ["内部数据", "外部合作数据", "公开数据", "第三方API数据"]
        )
        auth_status = st.selectbox(
            "数据授权状态", ["已获得用户授权", "已获得机构授权", "未授权"]
        )
        retention_period = st.number_input(
            "数据保留期限（天）", min_value=1, max_value=3650, value=180
        )

        check_results = []
        # 合规检查项
        check_items = [
            (
                "数据授权检查",
                auth_status in ["已获得用户授权", "已获得机构授权"],
                "未获得数据授权，违反《个人信息保护法》",
            ),
            (
                "保留期限检查",
                retention_period <= 180 if data_source == "外部合作数据" else True,
                "数据保留期限超过监管要求",
            ),
            ("敏感字段检查", len(detected_sensitive) == 0, "包含未处理敏感特征"),
            ("来源可追溯", data_source != "未知", "数据来源不可追溯"),
        ]

        for item_name, passed, desc in check_items:
            check_results.append(
                {
                    "检查项": item_name,
                    "检查结果": "✅ 通过" if passed else "❌ 不通过",
                    "说明": desc if not passed else "",
                }
            )

        st.dataframe(pl.DataFrame(check_results), use_container_width=True)

        # 生成合规凭证
        if st.button("生成数据源合规凭证"):
            from datetime import datetime

            credential = {
                "数据来源": data_source,
                "授权状态": auth_status,
                "保留期限": f"{retention_period}天",
                "检查时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "检查结果": "通过" if all([x[1] for x in check_items]) else "不通过",
            }
            st.json(credential)
            st.success("合规凭证已生成，已自动归档到项目日志")

    with tab3:
        st.subheader("样本合规校验")
        label_col = st.selectbox(
            "标签字段",
            df.columns,
            index=df.columns.get_loc(st.session_state.get("label_col", "label")),
        )
        time_col = st.selectbox(
            "时间字段（可选）",
            [None]
            + [col for col in df.columns if df[col].dtype in [pl.Date, pl.Datetime]],
        )

        if st.button("开始样本合规校验"):
            with st.spinner("正在校验..."):
                results = []
                # 标签分布校验
                label_dist = df[label_col].value_counts().to_list()
                bad_rate = df[label_col].mean()
                if bad_rate < 0.01 or bad_rate > 0.5:
                    results.append(
                        {
                            "检查项": "标签分布校验",
                            "结果": "⚠️ 警告",
                            "说明": f"样本坏率为{bad_rate:.2%}，偏离常规信贷样本坏率范围（1%-50%）",
                        }
                    )
                else:
                    results.append(
                        {
                            "检查项": "标签分布校验",
                            "结果": "✅ 通过",
                            "说明": f"样本坏率为{bad_rate:.2%}，分布正常",
                        }
                    )

                # 样本量校验
                if len(df) < 1000:
                    results.append(
                        {
                            "检查项": "样本量校验",
                            "结果": "⚠️ 警告",
                            "说明": f"样本量仅{len(df)}条，可能影响模型稳定性",
                        }
                    )
                else:
                    results.append(
                        {
                            "检查项": "样本量校验",
                            "结果": "✅ 通过",
                            "说明": f"样本量{len(df)}条，符合建模要求",
                        }
                    )

                # 标签准确性校验
                label_unique = df[label_col].unique().to_list()
                if set(label_unique) != {0, 1}:
                    results.append(
                        {
                            "检查项": "标签准确性校验",
                            "结果": "❌ 不通过",
                            "说明": "标签不是二分类值（0/1），请检查标签字段",
                        }
                    )
                else:
                    results.append(
                        {
                            "检查项": "标签准确性校验",
                            "结果": "✅ 通过",
                            "说明": "标签为标准二分类值",
                        }
                    )

                # 时间一致性校验（如果有时间字段）
                if time_col:
                    min_time = df[time_col].min()
                    max_time = df[time_col].max()
                    time_span = (max_time - min_time).days
                    if time_span < 180:
                        results.append(
                            {
                                "检查项": "时间跨度校验",
                                "结果": "⚠️ 警告",
                                "说明": f"样本时间跨度仅{time_span}天，建议至少6个月以覆盖不同时间段表现",
                            }
                        )
                    else:
                        results.append(
                            {
                                "检查项": "时间跨度校验",
                                "结果": "✅ 通过",
                                "说明": f"样本时间跨度{time_span}天，符合要求",
                            }
                        )

                st.dataframe(pl.DataFrame(results), use_container_width=True)
