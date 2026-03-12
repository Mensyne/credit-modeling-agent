import streamlit as st
import polars as pl
import pickle
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


def render_model_agent():
    st.title("🤖 模型Agent")

    if (
        "train_data" not in st.session_state
        or "selected_features" not in st.session_state
    ):
        st.warning("请先完成特征工程！")
        return

    train_df = st.session_state.train_data
    val_df = st.session_state.get("val_data")
    label_col = st.session_state.label_col
    feature_cols = st.session_state.selected_features

    st.write(f"训练集大小：{len(train_df)} 行，特征数：{len(feature_cols)}")
    if val_df is not None:
        st.write(f"验证集大小：{len(val_df)} 行")

    # 模型选择
    model_type = st.selectbox(
        "选择模型算法", ["逻辑回归", "XGBoost", "LightGBM", "CatBoost", "神经网络"]
    )

    # 模型默认参数
    model_params = {}

    if model_type == "逻辑回归":
        col1, col2 = st.columns(2)
        with col1:
            model_params["C"] = st.number_input("正则化强度C", value=1.0)
            model_params["max_iter"] = st.number_input("最大迭代次数", value=1000)
        with col2:
            model_params["penalty"] = st.selectbox(
                "正则化方式", ["l2", "l1", "elasticnet"], index=0
            )
            model_params["solver"] = st.selectbox(
                "求解器", ["lbfgs", "liblinear", "saga"], index=0
            )

    elif model_type in ["XGBoost", "LightGBM", "CatBoost"]:
        col1, col2, col3 = st.columns(3)
        with col1:
            model_params["n_estimators"] = st.number_input("树数量", value=100)
            model_params["max_depth"] = st.number_input("最大深度", value=3)
            model_params["learning_rate"] = st.number_input("学习率", value=0.1)
        with col2:
            model_params["subsample"] = st.slider("样本采样比例", 0.5, 1.0, 0.8)
            model_params["colsample_bytree"] = st.slider("特征采样比例", 0.5, 1.0, 0.8)
            model_params["min_child_weight"] = st.number_input(
                "最小叶子节点权重", value=1
            )
        with col3:
            model_params["random_state"] = st.number_input("随机种子", value=42)
            if model_type == "XGBoost":
                model_params["eval_metric"] = "auc"
            elif model_type == "LightGBM":
                model_params["metric"] = "auc"
                model_params["verbose"] = -1
            elif model_type == "CatBoost":
                model_params["verbose"] = 0
                model_params["eval_metric"] = "AUC"

    elif model_type == "神经网络":
        st.info("神经网络模型开发中，即将上线...")
        return

    # 训练按钮
    if st.button("开始训练模型", type="primary"):
        with st.spinner("正在训练模型..."):
            # 准备数据
            X_train = train_df.select(feature_cols).to_pandas()
            y_train = train_df[label_col].to_pandas()

            if val_df is not None:
                X_val = val_df.select(feature_cols).to_pandas()
                y_val = val_df[label_col].to_pandas()

            # 初始化模型
            if model_type == "逻辑回归":
                model = LogisticRegression(**model_params)
            elif model_type == "XGBoost":
                model = XGBClassifier(**model_params)
            elif model_type == "LightGBM":
                model = LGBMClassifier(**model_params)
            elif model_type == "CatBoost":
                model = CatBoostClassifier(**model_params)

            # 训练
            model.fit(X_train, y_train)

            # 预测
            y_train_pred = model.predict_proba(X_train)[:, 1]
            train_auc = roc_auc_score(y_train, y_train_pred)

            val_auc = None
            if val_df is not None:
                y_val_pred = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, y_val_pred)

            # 保存模型
            model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state.current_model = {
                "name": model_name,
                "type": model_type,
                "model": model,
                "params": model_params,
                "train_auc": train_auc,
                "val_auc": val_auc,
                "feature_cols": feature_cols,
                "label_col": label_col,
            }

            # 展示结果
            st.success("模型训练完成！")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("训练集AUC", f"{train_auc:.4f}")
            with col2:
                if val_auc is not None:
                    st.metric("验证集AUC", f"{val_auc:.4f}")

            # 特征重要性
            if hasattr(model, "feature_importances_"):
                st.subheader("特征重要性")
                importances = model.feature_importances_
                feat_imp = (
                    pl.DataFrame({"feature": feature_cols, "importance": importances})
                    .sort("importance", descending=True)
                    .head(20)
                )

                import plotly.express as px

                fig = px.bar(
                    feat_imp,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 20 特征重要性",
                )
                st.plotly_chart(fig, use_container_width=True)

    # 多模型对比
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = []

    if st.button("加入模型对比列表") and "current_model" in st.session_state:
        st.session_state.trained_models.append(st.session_state.current_model)
        st.success("已加入对比列表！")

    if len(st.session_state.trained_models) > 0:
        st.subheader("多模型对比")
        compare_data = []
        for model in st.session_state.trained_models:
            compare_data.append(
                {
                    "模型名称": model["name"],
                    "算法类型": model["type"],
                    "训练集AUC": f"{model['train_auc']:.4f}",
                    "验证集AUC": f"{model['val_auc']:.4f}" if model["val_auc"] else "-",
                }
            )
        st.dataframe(pl.DataFrame(compare_data), use_container_width=True)

        # 导出模型
        if st.button("导出选中模型") and "current_model" in st.session_state:
            model_path = f"./models/{st.session_state.current_model['name']}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(st.session_state.current_model, f)
            st.success(f"模型已导出到：{model_path}")
