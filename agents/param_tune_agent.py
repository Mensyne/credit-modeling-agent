import streamlit as st
import polars as pl
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import plotly.express as px


def render_param_tune_agent():
    st.title("⚙️ 参数调节Agent")

    if "current_model" not in st.session_state or "train_data" not in st.session_state:
        st.warning("请先训练初始模型！")
        return

    current_model = st.session_state.current_model
    train_df = st.session_state.train_data
    val_df = st.session_state.get("val_data")
    feature_cols = current_model["feature_cols"]
    label_col = current_model["label_col"]

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df[label_col].to_pandas()

    if val_df is not None:
        X_val = val_df.select(feature_cols).to_pandas()
        y_val = val_df[label_col].to_pandas()

    st.subheader(f"当前模型：{current_model['name']}（{current_model['type']}）")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("当前训练集AUC", f"{current_model['train_auc']:.4f}")
    with col2:
        if current_model["val_auc"]:
            st.metric("当前验证集AUC", f"{current_model['val_auc']:.4f}")

    tab1, tab2 = st.tabs(["🔧 手动调参", "🤖 自动调参"])

    with tab1:
        st.subheader("手动调参")
        model_type = current_model["type"]
        new_params = current_model["params"].copy()

        if model_type == "逻辑回归":
            col1, col2 = st.columns(2)
            with col1:
                new_params["C"] = st.slider(
                    "正则化强度C", 0.01, 10.0, value=new_params.get("C", 1.0)
                )
                new_params["max_iter"] = st.slider(
                    "最大迭代次数",
                    100,
                    5000,
                    value=new_params.get("max_iter", 1000),
                    step=100,
                )
            with col2:
                new_params["penalty"] = st.selectbox(
                    "正则化方式",
                    ["l2", "l1", "elasticnet"],
                    index=["l2", "l1", "elasticnet"].index(
                        new_params.get("penalty", "l2")
                    ),
                )
                new_params["solver"] = st.selectbox(
                    "求解器",
                    ["lbfgs", "liblinear", "saga"],
                    index=["lbfgs", "liblinear", "saga"].index(
                        new_params.get("solver", "lbfgs")
                    ),
                )

        elif model_type in ["XGBoost", "LightGBM", "CatBoost"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                new_params["n_estimators"] = st.slider(
                    "树数量",
                    50,
                    1000,
                    value=new_params.get("n_estimators", 100),
                    step=50,
                )
                new_params["max_depth"] = st.slider(
                    "最大深度", 1, 15, value=new_params.get("max_depth", 3)
                )
                new_params["learning_rate"] = st.slider(
                    "学习率",
                    0.001,
                    0.5,
                    value=new_params.get("learning_rate", 0.1),
                    step=0.001,
                )
            with col2:
                new_params["subsample"] = st.slider(
                    "样本采样比例",
                    0.5,
                    1.0,
                    value=new_params.get("subsample", 0.8),
                    step=0.05,
                )
                new_params["colsample_bytree"] = st.slider(
                    "特征采样比例",
                    0.5,
                    1.0,
                    value=new_params.get("colsample_bytree", 0.8),
                    step=0.05,
                )
                new_params["min_child_weight"] = st.slider(
                    "最小叶子节点权重",
                    1,
                    20,
                    value=new_params.get("min_child_weight", 1),
                )
            with col3:
                new_params["random_state"] = new_params.get("random_state", 42)

        if st.button("使用新参数训练", type="primary"):
            with st.spinner("正在训练..."):
                # 初始化模型
                if model_type == "逻辑回归":
                    model = LogisticRegression(**new_params)
                elif model_type == "XGBoost":
                    model = XGBClassifier(**new_params)
                elif model_type == "LightGBM":
                    model = LGBMClassifier(**new_params)
                elif model_type == "CatBoost":
                    model = CatBoostClassifier(**new_params, verbose=0)

                model.fit(X_train, y_train)

                # 评估
                y_train_pred = model.predict_proba(X_train)[:, 1]
                train_auc = roc_auc_score(y_train, y_train_pred)

                val_auc = None
                if val_df is not None:
                    y_val_pred = model.predict_proba(X_val)[:, 1]
                    val_auc = roc_auc_score(y_val, y_val_pred)

                # 保存模型
                st.session_state.current_model = {
                    "name": f"{model_type}_tuned_{pl.datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "type": model_type,
                    "model": model,
                    "params": new_params,
                    "train_auc": train_auc,
                    "val_auc": val_auc,
                    "feature_cols": feature_cols,
                    "label_col": label_col,
                }

                st.success("调参完成！")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "新训练集AUC",
                        f"{train_auc:.4f}",
                        f"{train_auc - current_model['train_auc']:.4f}",
                    )
                with col2:
                    if val_auc is not None:
                        st.metric(
                            "新验证集AUC",
                            f"{val_auc:.4f}",
                            f"{val_auc - current_model['val_auc']:.4f}",
                        )

    with tab2:
        st.subheader("自动调参")
        tune_method = st.selectbox("调参方法", ["贝叶斯优化", "随机搜索", "网格搜索"])
        n_trials = st.slider("迭代次数", 10, 100, value=20)
        metric = st.selectbox("优化指标", ["AUC", "accuracy", "f1"])

        if st.button("开始自动调参", type="primary"):
            with st.spinner("正在自动调参..."):
                model_type = current_model["type"]

                def objective(trial):
                    # 定义参数搜索空间
                    if model_type == "逻辑回归":
                        params = {
                            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                            "penalty": trial.suggest_categorical(
                                "penalty", ["l2", "l1"]
                            ),
                            "solver": "saga",
                            "max_iter": 1000,
                            "random_state": 42,
                        }
                        model = LogisticRegression(**params)

                    elif model_type == "XGBoost":
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                            "max_depth": trial.suggest_int("max_depth", 1, 10),
                            "learning_rate": trial.suggest_float(
                                "learning_rate", 0.01, 0.3, log=True
                            ),
                            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                            "colsample_bytree": trial.suggest_float(
                                "colsample_bytree", 0.6, 1.0
                            ),
                            "min_child_weight": trial.suggest_int(
                                "min_child_weight", 1, 10
                            ),
                            "random_state": 42,
                            "eval_metric": "auc",
                        }
                        model = XGBClassifier(**params)

                    elif model_type == "LightGBM":
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                            "max_depth": trial.suggest_int("max_depth", 1, 10),
                            "learning_rate": trial.suggest_float(
                                "learning_rate", 0.01, 0.3, log=True
                            ),
                            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                            "colsample_bytree": trial.suggest_float(
                                "colsample_bytree", 0.6, 1.0
                            ),
                            "min_child_weight": trial.suggest_int(
                                "min_child_weight", 1, 10
                            ),
                            "random_state": 42,
                            "verbose": -1,
                        }
                        model = LGBMClassifier(**params)

                    elif model_type == "CatBoost":
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                            "max_depth": trial.suggest_int("max_depth", 1, 10),
                            "learning_rate": trial.suggest_float(
                                "learning_rate", 0.01, 0.3, log=True
                            ),
                            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                            "colsample_bylevel": trial.suggest_float(
                                "colsample_bylevel", 0.6, 1.0
                            ),
                            "min_child_samples": trial.suggest_int(
                                "min_child_samples", 1, 20
                            ),
                            "random_state": 42,
                            "verbose": 0,
                        }
                        model = CatBoostClassifier(**params)

                    # 交叉验证
                    if val_df is not None:
                        model.fit(X_train, y_train)
                        y_pred = model.predict_proba(X_val)[:, 1]
                        score = roc_auc_score(y_val, y_pred)
                    else:
                        score = cross_val_score(
                            model, X_train, y_train, cv=5, scoring="roc_auc"
                        ).mean()
                    return score

                # 启动优化
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                # 展示结果
                best_params = study.best_params
                best_score = study.best_value

                st.success(f"调参完成！最佳{metric}：{best_score:.4f}")
                st.write("最佳参数：", best_params)

                # 调参过程可视化
                trials_df = pl.DataFrame(study.trials_dataframe())
                fig = px.line(
                    trials_df,
                    x="number",
                    y="value",
                    title="调参过程优化曲线",
                    labels={"number": "迭代次数", "value": "AUC值"},
                )
                st.plotly_chart(fig, use_container_width=True)

                # 参数重要性
                fig = optuna.visualization.plot_param_importances(study)
                st.plotly_chart(fig, use_container_width=True)

                # 训练最优模型
                if st.button("使用最优参数训练模型"):
                    with st.spinner("正在训练最优模型..."):
                        if model_type == "逻辑回归":
                            best_model = LogisticRegression(
                                **best_params, max_iter=1000, random_state=42
                            )
                        elif model_type == "XGBoost":
                            best_model = XGBClassifier(
                                **best_params, eval_metric="auc", random_state=42
                            )
                        elif model_type == "LightGBM":
                            best_model = LGBMClassifier(
                                **best_params, verbose=-1, random_state=42
                            )
                        elif model_type == "CatBoost":
                            best_model = CatBoostClassifier(
                                **best_params, verbose=0, random_state=42
                            )

                        best_model.fit(X_train, y_train)
                        y_train_pred = best_model.predict_proba(X_train)[:, 1]
                        train_auc = roc_auc_score(y_train, y_train_pred)

                        val_auc = None
                        if val_df is not None:
                            y_val_pred = best_model.predict_proba(X_val)[:, 1]
                            val_auc = roc_auc_score(y_val, y_val_pred)

                        # 保存模型
                        st.session_state.current_model = {
                            "name": f"{model_type}_auto_tuned_{pl.datetime.now().strftime('%Y%m%d%H%M%S')}",
                            "type": model_type,
                            "model": best_model,
                            "params": best_params,
                            "train_auc": train_auc,
                            "val_auc": val_auc,
                            "feature_cols": feature_cols,
                            "label_col": label_col,
                        }

                        st.success("最优模型训练完成！")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("训练集AUC", f"{train_auc:.4f}")
                        with col2:
                            if val_auc is not None:
                                st.metric("验证集AUC", f"{val_auc:.4f}")
