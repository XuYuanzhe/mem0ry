import json
import os
import uuid

# 设置目录路径
VECTOR_ID = str(uuid.uuid4())
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")
os.makedirs(mem0_dir, exist_ok=True)


def setup_config():
    config_path = os.path.join(mem0_dir, "config.json")
    if not os.path.exists(config_path):
        user_id = str(uuid.uuid4())
        config = {"user_id": user_id}
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)


def get_user_id():
    config_path = os.path.join(mem0_dir, "config.json")
    if not os.path.exists(config_path):
        return "anonymous_user"

    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            user_id = config.get("user_id")
            return user_id
    except Exception:
        return "anonymous_user"


def get_or_create_user_id(vector_store):
    """将user_id存储在向量数据库中并返回。"""
    user_id = get_user_id()

    # 尝试从向量数据库中获取现有的user_id
    try:
        existing = vector_store.get(vector_id=user_id)
        if existing and hasattr(existing, "payload") and existing.payload and "user_id" in existing.payload:
            return existing.payload["user_id"]
    except Exception:
        pass

    # 如果程序到达这里，我们需要插入user_id
    try:
        dims = getattr(vector_store, "embedding_model_dims", 1536)
        vector_store.insert(
            vectors=[[0.1] * dims], payloads=[{"user_id": user_id, "type": "user_identity"}], ids=[user_id]
        )
    except Exception:
        pass

    return user_id
