import streamlit as st
import sqlite3
import os
from datetime import datetime
from utils.auth_utils import hash_password, verify_password


# 初始化系统数据库
def init_system_db():
    db_path = os.getenv("DB_PATH", "./data/system.db")
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 用户表
    c.execute("""CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

    # 项目表
    c.execute("""CREATE TABLE IF NOT EXISTS projects
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  description TEXT,
                  owner_id INTEGER,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (owner_id) REFERENCES users(id))""")

    # 项目成员表
    c.execute("""CREATE TABLE IF NOT EXISTS project_members
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  project_id INTEGER,
                  user_id INTEGER,
                  role TEXT NOT NULL,
                  FOREIGN KEY (project_id) REFERENCES projects(id),
                  FOREIGN KEY (user_id) REFERENCES users(id),
                  UNIQUE(project_id, user_id))""")

    # 初始化管理员用户
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        hashed_pw = hash_password("admin123")
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", hashed_pw, "admin"),
        )

    conn.commit()
    conn.close()


def render_project_agent():
    st.title("📂 项目管理")

    # 初始化数据库
    init_system_db()

    # 登录状态检查
    if "user" not in st.session_state:
        st.subheader("用户登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        if st.button("登录"):
            db_path = os.getenv("DB_PATH", "./data/system.db")
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = c.fetchone()
            conn.close()

            if user and verify_password(password, user[2]):
                st.session_state.user = {
                    "id": user[0],
                    "username": user[1],
                    "role": user[3],
                }
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("用户名或密码错误")
        return

    # 已登录状态
    user = st.session_state.user
    st.sidebar.write(f"当前用户：{user['username']}（{user['role']}）")
    if st.sidebar.button("退出登录"):
        del st.session_state.user
        st.rerun()

    # 项目列表
    st.subheader("我的项目")
    db_path = os.getenv("DB_PATH", "./data/system.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 获取用户参与的项目
    c.execute(
        """SELECT p.id, p.name, p.description, p.created_at, pm.role 
                 FROM projects p
                 JOIN project_members pm ON p.id = pm.project_id
                 WHERE pm.user_id = ?
                 UNION
                 SELECT p.id, p.name, p.description, p.created_at, 'owner' as role
                 FROM projects p
                 WHERE p.owner_id = ?
                 ORDER BY p.created_at DESC""",
        (user["id"], user["id"]),
    )
    projects = c.fetchall()

    if projects:
        for proj in projects:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{proj[1]}**")
                st.write(f"{proj[2] or '无描述'}")
                st.write(f"创建时间：{proj[3]}，角色：{proj[4]}")
            with col2:
                if st.button("进入项目", key=f"enter_{proj[0]}"):
                    st.session_state.current_project = {
                        "id": proj[0],
                        "name": proj[1],
                        "role": proj[4],
                    }
                    st.success(f"已进入项目：{proj[1]}，即将跳转到数据Agent页面...")
                    # 自动跳转到数据Agent页面
                    st.session_state.page = "📥 数据Agent"
                    st.rerun()
            with col3:
                if proj[4] in ["owner", "admin"] and st.button(
                    "删除", key=f"del_{proj[0]}"
                ):
                    c.execute("DELETE FROM projects WHERE id = ?", (proj[0],))
                    c.execute(
                        "DELETE FROM project_members WHERE project_id = ?", (proj[0],)
                    )
                    conn.commit()
                    st.success("项目删除成功")
                    st.rerun()
            st.divider()
    else:
        st.info("暂无项目，点击下方创建新项目")

    # 创建新项目
    st.subheader("创建新项目")
    with st.form("create_project"):
        name = st.text_input("项目名称")
        description = st.text_area("项目描述")
        submit = st.form_submit_button("创建")

        if submit and name:
            c.execute(
                "INSERT INTO projects (name, description, owner_id) VALUES (?, ?, ?)",
                (name, description, user["id"]),
            )
            proj_id = c.lastrowid
            c.execute(
                "INSERT INTO project_members (project_id, user_id, role) VALUES (?, ?, ?)",
                (proj_id, user["id"], "owner"),
            )
            conn.commit()
            st.success("项目创建成功！")
            st.rerun()

    # 管理员功能：用户管理
    if user["role"] == "admin":
        st.divider()
        st.subheader("用户管理（管理员）")
        c.execute("SELECT id, username, role, created_at FROM users")
        users = c.fetchall()
        for u in users:
            st.write(f"ID: {u[0]}, 用户名: {u[1]}, 角色: {u[2]}, 创建时间: {u[3]}")

        # 新建用户
        with st.form("create_user"):
            new_username = st.text_input("新用户名")
            new_password = st.text_input("新密码", type="password")
            new_role = st.selectbox("角色", ["admin", "algorithm", "risk", "guest"])
            submit_user = st.form_submit_button("创建用户")

            if submit_user and new_username and new_password:
                hashed_pw = hash_password(new_password)
                try:
                    c.execute(
                        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                        (new_username, hashed_pw, new_role),
                    )
                    conn.commit()
                    st.success("用户创建成功")
                except sqlite3.IntegrityError:
                    st.error("用户名已存在")

    conn.close()
