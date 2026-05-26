import streamlit as st
import pickle
import os
from datetime import datetime
import json


def render_deploy_agent():
    st.title("🚀 模型部署Agent")

    if "current_model" not in st.session_state:
        st.warning("请先训练模型！")
        return

    model = st.session_state.current_model
    st.subheader(f"当前模型：{model['name']}（{model['type']}）")

    tab1, tab2, tab3 = st.tabs(["📦 模型导出", "⚡ 一键部署", "📄 接口文档"])

    with tab1:
        st.subheader("模型导出")
        export_format = st.multiselect(
            "选择导出格式",
            ["Pickle (.pkl)", "ONNX (.onnx)", "PMML (.pmml)"],
            default=["Pickle (.pkl)"],
        )

        if st.button("开始导出", type="primary"):
            with st.spinner("正在导出模型..."):
                export_results = []
                os.makedirs("./models", exist_ok=True)
                base_name = f"{model['name']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

                if "Pickle (.pkl)" in export_format:
                    pickle_path = f"./models/{base_name}.pkl"
                    with open(pickle_path, "wb") as f:
                        pickle.dump(model, f)
                    export_results.append(
                        {
                            "格式": "Pickle",
                            "路径": pickle_path,
                            "大小": f"{os.path.getsize(pickle_path) / 1024 / 1024:.2f} MB",
                        }
                    )

                if "ONNX (.onnx)" in export_format:
                    try:
                        import onnxmltools
                        from skl2onnx.common.data_types import FloatTensorType

                        X_test = [[0.0] * len(model["feature_cols"])]
                        initial_type = [
                            (
                                "float_input",
                                FloatTensorType([None, len(model["feature_cols"])]),
                            )
                        ]
                        onnx_model = onnxmltools.convert_sklearn(
                            model["model"], initial_types=initial_type
                        )
                        onnx_path = f"./models/{base_name}.onnx"
                        onnxmltools.utils.save_model(onnx_model, onnx_path)
                        export_results.append(
                            {
                                "格式": "ONNX",
                                "路径": onnx_path,
                                "大小": f"{os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB",
                            }
                        )
                    except Exception as e:
                        st.warning(f"ONNX导出失败：{str(e)}，当前模型暂不支持ONNX格式")

                if "PMML (.pmml)" in export_format:
                    try:
                        from sklearn2pmml import sklearn2pmml, PMMLPipeline
                        from sklearn.pipeline import Pipeline

                        pmml_pipeline = PMMLPipeline([("classifier", model["model"])])
                        pmml_path = f"./models/{base_name}.pmml"
                        sklearn2pmml(pmml_pipeline, pmml_path, with_repr=True)
                        export_results.append(
                            {
                                "格式": "PMML",
                                "路径": pmml_path,
                                "大小": f"{os.path.getsize(pmml_path) / 1024 / 1024:.2f} MB",
                            }
                        )
                    except Exception as e:
                        st.warning(f"PMML导出失败：{str(e)}，当前模型暂不支持PMML格式")

                st.success("模型导出完成！")
                if export_results:
                    st.dataframe(export_results, use_container_width=True)

                    # 下载按钮
                    for res in export_results:
                        with open(res["路径"], "rb") as f:
                            st.download_button(
                                label=f"下载{res['格式']}文件",
                                data=f.read(),
                                file_name=os.path.basename(res["路径"]),
                                mime="application/octet-stream",
                            )

    with tab2:
        st.subheader("一键部署API服务")
        deploy_port = st.number_input("部署端口", value=8000)
        enable_auth = st.checkbox("启用接口鉴权", value=True)

        if st.button("生成部署包"):
            with st.spinner("正在生成部署包..."):
                # 生成FastAPI部署代码
                deploy_code = f"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import pickle
import numpy as np
from pydantic import BaseModel
from typing import List

app = FastAPI(title="{model["name"]} 推理接口", version="1.0")
security = HTTPBearer()

BASE_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_DIR, "{base_name}.pkl")

# 加载模型
with open(MODEL_FILE, "rb") as f:
    model_data = pickle.load(f)

MODEL = model_data["model"]
FEATURE_COLS = {json.dumps(model["feature_cols"])}
SECRET_KEY = os.getenv("DEPLOY_SECRET_KEY", "your-secret-key-here")

# 请求模型
class PredictRequest(BaseModel):
    features: List[float]

# 鉴权依赖
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return credentials

@app.post("/predict", summary="模型预测接口")
async def predict(request: PredictRequest, auth: HTTPAuthorizationCredentials = Depends(verify_token)):
    if len(request.features) != len(FEATURE_COLS):
        raise HTTPException(status_code=400, detail=f"特征数量不匹配，需要{len(FEATURE_COLS)}个特征")

    features = np.array(request.features).reshape(1, -1)
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(features)[0][1]
    else:
        pred = MODEL.predict(features)[0]
        proba = float(pred)

    return {{
        "probability": float(proba),
        "result": "高风险" if proba >= 0.5 else "低风险",
        "threshold": 0.5
    }}

@app.get("/health", summary="健康检查接口")
async def health():
    return {{"status": "ok", "model": "{model["name"]}"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port={deploy_port})
"""
                # 保存部署代码
                deploy_path = f"./models/{base_name}_deploy.py"
                with open(deploy_path, "w", encoding="utf-8") as f:
                    f.write(deploy_code)

                # 保存requirements
                requirements = """fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.2
python-multipart==0.0.6
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
catboost==1.2.0
numpy==1.26.2
"""
                req_path = f"./models/{base_name}_requirements.txt"
                with open(req_path, "w", encoding="utf-8") as f:
                    f.write(requirements)

                st.success("部署包生成完成！")
                st.code(deploy_code, language="python")

                st.info(f"""
                部署命令：
                ```bash
                pip install -r {os.path.basename(req_path)}
                python {os.path.basename(deploy_path)}
                ```
                接口地址：http://localhost:{deploy_port}
                接口文档地址：http://localhost:{deploy_port}/docs
                """)

    with tab3:
        st.subheader("接口文档")
        st.markdown("### 1. 预测接口 /predict")
        st.markdown("**请求方式：** POST")
        st.markdown("**请求头：** Authorization: Bearer <your-token>")
        st.markdown("**请求参数：**")
        st.json({"features": [float(x) for x in range(len(model["feature_cols"]))]})
        st.markdown("**响应参数：**")
        st.json({"probability": 0.75, "result": "高风险", "threshold": 0.5})

        st.markdown("### 2. 健康检查接口 /health")
        st.markdown("**请求方式：** GET")
        st.markdown("**响应参数：**")
        st.json({"status": "ok", "model": model["name"]})

        st.markdown("### 3. 特征说明")
        st.dataframe(
            st.dataframe(
                {
                    "特征序号": range(1, len(model["feature_cols"]) + 1),
                    "特征名": model["feature_cols"],
                }
            ),
            use_container_width=True,
        )
