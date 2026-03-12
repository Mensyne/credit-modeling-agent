import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()


def hash_password(password: str) -> str:
    """哈希密码"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def check_permission(user_role: str, required_role: str) -> bool:
    """检查用户权限
    权限层级：admin > algorithm > risk > guest
    """
    role_hierarchy = {"guest": 1, "risk": 2, "algorithm": 3, "admin": 4}
    return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)
