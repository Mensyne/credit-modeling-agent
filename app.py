import streamlit as st
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化页面状态
if "page" not in st.session_state:
    st.session_state.page = "🏠 首页"

# 页面配置
st.set_page_config(
    page_title="信贷建模Agent可视化系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义样式
css_path = os.path.join(os.path.dirname(__file__), "assets", "css", "custom.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("未找到自定义样式文件 assets/css/custom.css，已使用默认样式。")

# 导航菜单
menu = [
    "🏠 首页",
    "📂 项目管理",
    "📥 数据Agent",
    "✂️ 样本划分Agent",
    "🔧 特征工程Agent",
    "🤖 模型Agent",
    "⚙️ 参数调节Agent",
    "📈 模型效果Agent",
    "⚡ 模型性能Agent",
    "🎨 可视化Agent",
    "✅ 数据合规Agent",
    "🔍 可解释性Agent",
    "📑 监管合规报告Agent",
    "📄 模型报告Agent",
    "🚀 模型部署Agent",
    "📡 线上监控Agent",
]

# 侧边栏导航
choice = st.sidebar.selectbox("导航菜单", menu, index=menu.index(st.session_state.page))
# 同步页面状态
if choice != st.session_state.page:
    st.session_state.page = choice
    st.rerun()

# 页面路由
if choice == "🏠 首页":
    st.title("📊 信贷建模Agent可视化系统")
    st.markdown("### 全流程信贷建模自动化平台")
    st.markdown("""
    本系统覆盖**数据接入→样本划分→特征工程→模型训练→调参→评估→合规→部署→监控**全链路，
    支持多角色协作、多项目隔离，满足信贷监管合规要求，大幅提升建模效率。
    
    #### 核心功能：
    - 🚀 高性能数据处理：基于Polars实现，100万行数据加载<3s
    - 🤖 多算法支持：逻辑回归/XGBoost/LightGBM/CatBoost/NN/图学习
    - 📊 专属可视化：特征分箱、模型效果、分数分布多维度可视化
    - ✅ 合规内置：自动识别敏感特征、生成监管要求报告
    - 🔒 权限管理：多角色多项目隔离，操作全留痕
    """)

elif choice == "📂 项目管理":
    from agents.project_agent import render_project_agent

    render_project_agent()

elif choice == "📥 数据Agent":
    from agents.data_agent import render_data_agent

    render_data_agent()

elif choice == "✂️ 样本划分Agent":
    from agents.sample_split_agent import render_sample_split_agent

    render_sample_split_agent()

elif choice == "🔧 特征工程Agent":
    from agents.feature_agent import render_feature_agent

    render_feature_agent()

elif choice == "🤖 模型Agent":
    from agents.model_agent import render_model_agent

    render_model_agent()

elif choice == "⚙️ 参数调节Agent":
    from agents.param_tune_agent import render_param_tune_agent

    render_param_tune_agent()

elif choice == "📈 模型效果Agent":
    from agents.effect_agent import render_effect_agent

    render_effect_agent()

elif choice == "⚡ 模型性能Agent":
    from agents.performance_agent import render_performance_agent

    render_performance_agent()

elif choice == "🎨 可视化Agent":
    from agents.visualization_agent import render_visualization_agent

    render_visualization_agent()

elif choice == "✅ 数据合规Agent":
    from agents.compliance_agent import render_compliance_agent

    render_compliance_agent()

elif choice == "🔍 可解释性Agent":
    from agents.interpret_agent import render_interpret_agent

    render_interpret_agent()

elif choice == "📑 监管合规报告Agent":
    from agents.regulatory_report_agent import render_regulatory_report_agent

    render_regulatory_report_agent()

elif choice == "📄 模型报告Agent":
    from agents.model_report_agent import render_model_report_agent

    render_model_report_agent()

elif choice == "🚀 模型部署Agent":
    from agents.deploy_agent import render_deploy_agent

    render_deploy_agent()

elif choice == "📡 线上监控Agent":
    from agents.monitor_agent import render_monitor_agent

    render_monitor_agent()

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 信贷建模Agent系统 v1.0.0")
