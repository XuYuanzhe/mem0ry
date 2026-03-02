import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pytz
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from mem0.exceptions import ValidationError as Mem0ValidationError
from mem0.memory.base import MemoryBase
from mem0.memory.setup import mem0_dir, setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import (
    extract_json,
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from mem0.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    VectorStoreFactory,
    RerankerFactory,
)

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)


def _safe_deepcopy_config(config):
    """安全地深度复制配置，对不可序列化的对象采用JSON序列化。"""
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using JSON serialization: {e}")

        config_class = type(config)

        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump(mode="json")
            except Exception:
                clone_dict = {k: v for k, v in config.__dict__.items()}
        elif hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict
            clone_dict = asdict(config)
        else:
            clone_dict = {k: v for k, v in config.__dict__.items()}

        sensitive_tokens = ("auth", "credential", "password", "token", "secret", "key", "connection_class")
        for field_name in list(clone_dict.keys()):
            if any(token in field_name.lower() for token in sensitive_tokens):
                clone_dict[field_name] = None

        try:
            return config_class(**clone_dict)
        except Exception as reconstruction_error:
            logger.warning(
                f"Failed to reconstruct config: {reconstruction_error}. "
                f"Telemetry may be affected."
            )
            raise


def _build_filters_and_metadata(
        *,  # 强制关键字参数，* 之后的参数必须显式地写成关键字参数（Keyword-Only），调用时不能按位置传递。这里设计严重的数据安全问题，所以添加了 *
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        actor_id: Optional[str] = None,  # For query-time filtering
        input_metadata: Optional[Dict[str, Any]] = None,
        input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    构建用于存储的元数据以及基于会话和参与者标识符的查询过滤器。

    此辅助函数支持多个会话标识符（user_id、agent_id 和/或 run_id），以实现灵活的会话范围界定，并可选择性地将查询缩小到特定的 actor_id。它返回两个字典：

    1、base_metadata_template：用作存储新记忆时的元数据模板。它包含所有提供的会话标识符以及任何 input_metadata。
    2、effective_query_filters：用作查询现有记忆的过滤器。它包含所有提供的会话标识符、任何 input_filters，以及如果指定了任何与参与者相关的输入，则包含一个已解析的参与者标识符以进行针对性过滤。

    参与者过滤优先级：显式的 actor_id 参数 → filters["actor_id"]。
    此已解析的参与者 ID 仅用于查询，不会添加到 base_metadata_template 中，因为存储时的参与者通常会在后续阶段从消息内容中推导得出。

    参数：
        user_id (Optional[str])：用户标识符，用于会话范围界定。
        agent_id (Optional[str])：代理标识符，用于会话范围界定。
        run_id (Optional[str])：运行标识符，用于会话范围界定。
        actor_id (Optional[str])：显式的参与者标识符，用作特定参与者过滤的潜在来源。请参阅主描述中的参与者解析优先级。
        input_metadata (Optional[Dict[str, Any]])：基础字典，将用会话标识符进行增强，以生成存储元数据模板。默认为空字典。
        input_filters (Optional[Dict[str, Any]])：基础字典，将用会话和参与者标识符进行增强，以生成查询过滤器。默认为空字典。

    返回：
        tuple[Dict[str, Any], Dict[str, Any]]：包含以下内容的元组：
            base_metadata_template (Dict[str, Any])：用于存储记忆的元数据模板，限定于提供的会话范围。
            effective_query_filters (Dict[str, Any])：用于查询记忆的过滤器，限定于提供的会话范围，并可能包含已解析的参与者。
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    # 添加所有的 id 到 session_ids_provided
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    # 如果没有任何一个 id 则抛异常
    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    # 筛选 actor_id
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


setup_config()
logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        # 自定义事实提取 Prompt
        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        # 自定义更新内存 Prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        # 嵌入模型
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        # 向量数据库（默认 qdrant 在 config 中可以通过 vector_store 配置）
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        # LLM（默认 openai 在 config 中可以通过 llm 配置）
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        # 历史数据库（默认 sqlite 在 config 中可以通过 history_db_path 配置存储路径，但不支持通过 config 修改）
        self.db = SQLiteManager(self.config.history_db_path)
        # 向量数据库表名（存业务数据）
        self.collection_name = self.config.vector_store.config.collection_name
        # 目前支持 v1.0 和 v1.1 两个版本
        self.api_version = self.config.version

        # 如果配置了重排序器，则初始化它
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # 图数据库配置（支持在 config 中通过 graph_store 配置）
        self.enable_graph = False

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            self.graph = GraphStoreFactory.create(provider, self.config)
            self.enable_graph = True
        else:
            self.graph = None
        # 手动创建遥测配置字典，避免线程锁的深度复制问题（存内部监控数据）
        telemetry_config_dict = {}
        if hasattr(self.config.vector_store.config, 'model_dump'):
            # For pydantic models
            telemetry_config_dict = self.config.vector_store.config.model_dump()
        else:
            # 对于其他对象，手动复制常用属性
            for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                if hasattr(self.config.vector_store.config, attr):
                    telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

        # 覆盖遥测数据库表名
        telemetry_config_dict['collection_name'] = "mem0migrations"

        # 设置基于文件的向量存储的路径
        telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
        if self.config.vector_store.provider in ["faiss", "qdrant"]:
            provider_path = f"migrations_{self.config.vector_store.provider}"
            telemetry_config_dict['path'] = os.path.join(mem0_dir, provider_path)
            os.makedirs(telemetry_config_dict['path'], exist_ok=True)

        # 使用与原始类相同的类创建配置对象
        telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
        self._telemetry_vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, telemetry_config
        )
        # 用于捕获事件（可忽略）
        capture_event("mem0.init", self, {"sync_type": "sync"})

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def add(
            self,
            messages,
            *,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            infer: bool = True,
            memory_type: Optional[str] = None,
            prompt: Optional[str] = None,
    ):
        """
        创建新记忆。

        此功能用于添加限定于单个会话 ID（例如 user_id、agent_id 或 run_id）的新记忆。必须提供上述 ID 中的至少一个。

        参数：
            messages (str 或 List[Dict[str, str]])：要处理和存储的消息内容或消息列表（例如，[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]）。
            user_id (str, 可选)：创建记忆的用户 ID。默认为 None。
            agent_id (str, 可选)：创建记忆的代理 ID。默认为 None。
            run_id (str, 可选)：创建记忆的运行 ID。默认为 None。
            metadata (dict, 可选)：与记忆一起存储的元数据。默认为 None。
            infer (bool, 可选)：如果为 True（默认值），则使用大语言模型 (LLM) 从 'messages' 中提取关键事实，并决定是添加、更新还是删除相关记忆。如果为 False，'messages' 将作为原始记忆直接添加。
            memory_type (str, 可选)：指定记忆类型。目前，仅明确处理 MemoryType.PROCEDURAL.value ("procedural_memory") 以创建过程性记忆（通常需要 'agent_id'）。否则，记忆被视为一般对话/事实记忆。默认为 None。默认情况下，它会创建短期记忆以及长期记忆（语义记忆和情景记忆）。传入 "procedural_memory" 可创建过程性记忆。
            prompt (str, 可选)：用于创建记忆的提示词。默认为 None。

        返回：
            dict：包含记忆添加操作结果的字典，通常在 "results" 键下包含受影响的记忆项列表（已添加、已更新），如果启用了图存储，还可能包含 "relations"。
            v1.1+ 示例：{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}

        异常：
            Mem0ValidationError：如果输入验证失败（无效的 memory_type、messages 格式等）。
            VectorStoreError：如果向量存储操作失败。
            GraphStoreError：如果图存储操作失败。
            EmbeddingError：如果嵌入生成失败。
            LLMError：如果 LLM 操作失败。
            DatabaseError：如果数据库操作失败。
        """

        # 存储新记忆的 dict 、查询记忆的 dict
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        # 记忆类型若不是程序性记忆（PROCEDURAL）抛出异常（原因见参数注释）
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        # 处理输入消息（必须为 str, dict, list 中的一种）
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        # 程序性记忆（PROCEDURAL）要配合 agent_id 才能创建
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results

        # 如果配置了多模态支持图片
        if self.config.llm.config.get("enable_vision"):
            # 则按照配置将包含图片的“多模态消息”转换为纯文本消息
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            # 将包含图片的“多模态消息”转换为纯文本消息
            messages = parse_vision_messages(messages)

        # 并行处理，
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 任务 1：存入向量数据库 (Vector Store) 如果 infer=True，则会调用 LLM 进行语义分析、提取实体、分类记忆类型。
            future1 = executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            # 任务 2：存入知识图谱 (Graph Store) 用于关系推理和结构化查询
            future2 = executor.submit(self._add_to_graph, messages, effective_filters)

            # 等待两个任务都做完
            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()  # 获取向量存储的返回结果
            graph_result = future2.result()  # 获取图谱存储的返回结果

        # 如果配置了知识图谱则返回：向量存储的返回结果、图谱存储的返回结果
        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        # 如果没有配置知识图谱返回：向量存储的返回结果
        return {"results": vector_store_result}

    # 记忆生命周期管理引擎（核心能力之一、大脑决策层）
    def _add_to_vector_store(self, messages, metadata, filters, infer):
        # P1、不需要推理直接存储
        if not infer:
            # 不做任何 AI 分析，不提取事实。直接把用户说的话当作记忆存进去
            returned_memories = []
            # 直接遍历消息，生成 Embedding，存入数据库
            for message_dict in messages:
                if (
                        not isinstance(message_dict, dict)
                        or message_dict.get("role") is None
                        or message_dict.get("content") is None
                ):
                    logger.warning(f"跳过无效的消息格式: {message_dict}")
                    continue

                # 忽略系统角色
                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                # 创建记忆
                mem_id = self._create_memory(msg_content, msg_embeddings, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # P2、提取实时（AI 推理）
        # 1. 解析消息
        parsed_messages = parse_messages(messages)

        # 2. 构建 Prompt (支持自定义)
        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            # 自动判断是普通用户记忆还是 Agent 记忆，选择不同 Prompt
            is_agent_memory = self._should_use_agent_memory_extraction(messages, metadata)
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)

        # 3. 调用 LLM 提取事实 (JSON 格式)
        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        # 使用 new_retrieved_facts 来接收提取到的事实列表
        try:
            # 移除代码块
            response = remove_code_blocks(response)
            if not response.strip():
                new_retrieved_facts = []
            else:
                try:
                    # 尝试直接JSON解析
                    new_retrieved_facts = json.loads(response)["facts"]
                except json.JSONDecodeError:
                    # 尝试使用内置函数从响应中提取JSON
                    extracted_json = extract_json(response)
                    new_retrieved_facts = json.loads(extracted_json)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        if not new_retrieved_facts:
            logger.debug("未从输入中检索到新事实。跳过 memory update LLM call。")

        retrieved_old_memory = []
        new_message_embeddings = {}
        # Search for existing memories using the provided session identifiers
        # Use all available session identifiers for accurate memory retrieval
        search_filters = {}
        if filters.get("user_id"):
            search_filters["user_id"] = filters["user_id"]
        if filters.get("agent_id"):
            search_filters["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            search_filters["run_id"] = filters["run_id"]

        # P3、检索与去重（防止记忆冗余）
        for new_mem in new_retrieved_facts:
            # 1. 生成新事实的向量
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            # 2. 在数据库中搜索相似记忆 (Top 5 条)
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=search_filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        # P4、决策与规划（LLM 作为大脑）
        if new_retrieved_facts:
            # 1. 构建 Prompt：把 [旧记忆] 和 [新事实] 一起给 LLM
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )

            # 2. 调用 LLM 做决策
            try:
                response: str = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""

            try:
                if not response or not response.strip():
                    logger.warning("LLM的响应为空，没有可提取的记忆")
                    new_memories_with_actions = {}
                else:
                    response = remove_code_blocks(response)
                    new_memories_with_actions = json.loads(response)
            except Exception as e:
                logger.error(f"JSON 响应无效: {e}")
                new_memories_with_actions = {}
        else:
            # 这个分支代表 P2 没有提取到新的事实
            new_memories_with_actions = {}

        # P5、执行操作（CRUD 落地）
        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        logger.info("Skipping memory entry because of empty `text` field.")
                        continue

                    event_type = resp.get("event")  # ADD, UPDATE, DELETE, NONE
                    if event_type == "ADD":
                        # 创建新记忆
                        memory_id = self._create_memory(
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                    elif event_type == "UPDATE":
                        # 更新旧向量内容和 Embedding
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp.get("id")],
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        # 物理删除
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "NONE":
                        # 即使内容不变，也可能需要更新 metadata (如 session_id)
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                            # 仅更新会话标识符，保持内容不变
                            existing_memory = self.vector_store.get(vector_id=memory_id)
                            updated_metadata = deepcopy(existing_memory.payload)
                            if metadata.get("agent_id"):
                                updated_metadata["agent_id"] = metadata["agent_id"]
                            if metadata.get("run_id"):
                                updated_metadata["run_id"] = metadata["run_id"]
                            updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

                            self.vector_store.update(
                                vector_id=memory_id,
                                vector=None,  # 保持相同的 embeddings
                                payload=updated_metadata,
                            )
                            logger.info(f"Updated session IDs for memory {memory_id}")
                        else:
                            logger.info("NOOP for Memory.")
                except Exception as e:
                    logger.error(f"Error processing memory action: {resp}, Error: {e}")
        except Exception as e:
            logger.error(f"迭代 new_memories_with_actions 时出错: {e}")

        keys, encoded_ids = process_telemetry_filters(filters)
        # 用于捕获事件（可忽略）
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        """
        通过 ID 检索记忆。

        参数：
            memory_id (str)：要检索的记忆的 ID。

        返回：
            dict：检索到的记忆。
        """
        # 用于捕获事件（可忽略）
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
            self,
            *,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 100,
    ):
        """
        查询所有记忆列表。

        参数：
            user_id (str, 可选)：用户 ID。
            agent_id (str, 可选)：代理 ID。
            run_id (str, 可选)：运行 ID。
            filters (dict, 可选)：应用于搜索的额外自定义键值过滤器。这些过滤器会与基于 ID 的范围界定过滤器合并。例如，filters={"actor_id": "some_user"}。
            limit (int, 可选)：要返回的最大记忆数量。默认为 100。
        返回：
            dict：一个字典，在 "results" 键下包含记忆列表；如果启用了图存储，还可能包含 "relations"。对于 API v1.0，它可能直接返回一个列表（参见弃用警告）。
            v1.1+ 示例：{"results": [{"id": "...", "memory": "...", ...}]}
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("必须至少指定“user_id”、“agent_id”或“run_id”中的一个.")

        keys, encoded_ids = process_telemetry_filters(effective_filters)

        # 用于捕获事件（可忽略）
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}

        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, limit=limit)

        # 通过检查第一个元素来处理不同的向量存储返回格式。
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # 如果第一个元素是容器，则解包一层。
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # 第一个元素是记忆对象，结构已经是扁平的。
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
            self,
            query: str,
            *,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            limit: int = 100,
            filters: Optional[Dict[str, Any]] = None,
            threshold: Optional[float] = None,
            rerank: bool = True,
    ):
        """
        根据查询搜索记忆。
        参数：
            query (str)：要搜索的查询内容。
            user_id (str, 可选)：要搜索的用户 ID。默认为 None。
            agent_id (str, 可选)：要搜索的代理 ID。默认为 None。
            run_id (str, 可选)：要搜索的运行 ID。默认为 None。
            limit (int, 可选)：限制结果数量。默认为 100。
            filters (dict, 可选)：应用于搜索的旧版过滤器。默认为 None。
            threshold (float, 可选)：记忆被包含在结果中的最低分数阈值。默认为 None。
            filters (dict, 可选)：增强的元数据过滤，支持以下运算符：
                {"key": "value"} - 精确匹配
                {"key": {"eq": "value"}} - 等于
                {"key": {"ne": "value"}} - 不等于
                {"key": {"in": ["val1", "val2"]}} - 在列表中
                {"key": {"nin": ["val1", "val2"]}} - 不在列表中
                {"key": {"gt": 10}} - 大于
                {"key": {"gte": 10}} - 大于或等于
                {"key": {"lt": 10}} - 小于
                {"key": {"lte": 10}} - 小于或等于
                {"key": {"contains": "text"}} - 包含文本
                {"key": {"icontains": "text"}} - 不区分大小写的包含
                {"key": "*"} - 通配符匹配（任意值）
                {"AND": [filter1, filter2]} - 逻辑与
                {"OR": [filter1, filter2]} - 逻辑或
                {"NOT": [filter1]} - 逻辑非
        返回：
            dict：包含搜索结果的字典，通常在 "results" 键下；如果启用了图存储，还可能包含 "relations"。
            v1.1+ 示例：{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}
        """
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("必须至少指定“user_id”、“agent_id”或“run_id”中的一个。")

        # 如果检测到高级运算符，则应用增强的元数据过滤
        if filters and self._has_advanced_operators(filters):
            processed_filters = self._process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # 简单的过滤器，直接合并
            effective_filters.update(filters)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 用于捕获事件（可忽略）
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 粗排序
            # 优点：速度极快（毫秒级）
            # 缺点：只能理解“语义相似”，容易受关键词干扰或忽略逻辑关系。
            future_memories = executor.submit(self._search_vector_store, query, effective_filters, limit, threshold)
            future_graph_entities = (
                executor.submit(self.graph.search, query, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            original_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        # 精排序（如果设置了则重排序）
        # 优点：精度极高，能理解复杂的逻辑、否定句和上下文依赖。
        # 缺点：速度较慢（因为要计算 N 次），所以通常只对粗排回来的 Top-K（比如前 50 或 100 条）进行重排序。
        if rerank and self.reranker and original_memories:
            try:
                # 将查询（Query）和召回的每条记忆（Document）成对地输入到 config 中配置好的 Reranker 模型中。
                # 这个模型会深度分析两者之间的逻辑匹配度，并给出一个全新的、更精准的分数。
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                original_memories = reranked_memories
            except Exception as e:
                # 容错处理
                logger.warning(f"重排序失败, 使用原始结果: {e}")

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        return {"results": original_memories}

    @staticmethod
    def _process_metadata_filters(metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理增强的元数据过滤器，并将其转换为向量存储兼容的格式。

        参数：
            metadata_filters：带有运算符的增强型元数据过滤器。

        返回：
            Dict：与向量存储兼容的处理后过滤器字典。
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result[key] = {operator_map[operator]: value}
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))

        return processed_filters

    @staticmethod
    def _has_advanced_operators(filters: Dict[str, Any]) -> bool:
        """
        检查过滤器是否包含需要特殊处理的高级运算符。

        参数：
            filters：要检查的过滤器字典。

        返回：
            bool：如果检测到高级运算符，则返回 True。
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        # 将查询条件 Query 转化为向量（通过 config 中 配置的 embedding_model 模型实现）
        embeddings = self.embedding_model.embed(query, "search")

        # 在向量数据库中计算它与成千上万条记忆的余弦相似度
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            # 确保只返回相似度高于设定值的记忆
            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    def update(self, memory_id, data):
        """
        通过 ID 更新记忆。

        参数：
            memory_id (str)：要更新的记忆 ID。
            data (str)：用于更新记忆的新内容。

        返回：
            dict：指示记忆已成功更新的 success 消息。

        示例:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        # 用于捕获事件（可忽略）
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        # 用于捕获事件（可忽略）
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        删除所有记忆。

        参数：
            user_id (str, 可选)：要删除记忆的用户 ID。默认为 None。
            agent_id (str, 可选)：要删除记忆的代理 ID。默认为 None。
            run_id (str, 可选)：要删除记忆的运行 ID。默认为 None。
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "至少需要一个ID过滤器来删除所有内存。如果你想删除所有内存，请使用`reset（）`方法。"
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        # 用于捕获事件（可忽略）
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # 删除所有向量存储器并重置集合
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)
        self.vector_store.reset()

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        获取指定 ID 记忆的变更历史。

        参数：
            memory_id (str)：要获取历史的记忆 ID。

        返回：
            list：该记忆的变更列表。
        """
        # 用于捕获事件（可忽略）
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        return self.db.get_history(memory_id)

    # 创建记忆（执行层）
    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # 执行向量数据库新增操作
        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )

        # 记录历史审计日志（Audit Trail）
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        # 用于捕获事件（可忽略）
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    # 更新记忆（执行层）
    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        """
        如果把记忆更新比作“修改档案”：
        1. 大脑决策层 (_add_to_vector_store): LLM 分析后说：“嘿，那条关于‘喜欢咖啡’的记忆过时了，用户现在喜欢茶了，我们需要执行 UPDATE 操作，目标 ID 是 123。”
        2.执行层 (_update_memory):
            走到档案柜（向量库）。
            抽出 123 号档案，复印一份留底（记录 prev_value）。
            把里面的文字改成“喜欢茶”。
            重新计算索引标签（生成新 embeddings）。
            把新档案塞回去（vector_store.update）。
            在登记本上写一笔：“123 号档案于 10:26 由‘喜欢咖啡’改为‘喜欢茶’”（add_history）。
        """
        logger.info(f"Updating memory with {data=}")

        # P1、读取旧数据（用于审计/历史）
        try:
            # 为了记录“修改前”的内容，以便后续写入历史日志（History Log），实现记忆的可追溯性。
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"在更新过程中获取ID为{memory_id}的内存时出错.")
            raise ValueError(f"获取ID为 ｛memory_id｝ 的内存时出错。请提供有效的 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        # P2、构建新元数据（Metadata Merge）
        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # 保留旧的 session ID (user_id, agent_id 等)，除非新元数据里显式提供了新的
        # 确保更新后的记忆依然属于原来的用户（user_id）或代理（agent_id）。如果不小心把这些 ID 弄丢了，这条记忆就变成孤儿数据，再也查不到。
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" not in new_metadata and "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        # P3、生成新向量（Embedding）
        if data in existing_embeddings:
            # 如果在之前的步骤中已经计算过这段新文本的向量，就直接复用（existing_embeddings），避免重复调用 AI 接口，节省时间和成本。
            embeddings = existing_embeddings[data]
        else:
            # 如果没有，则实时调用嵌入模型生成新的向量。
            embeddings = self.embedding_model.embed(data, "update")

        # P4、执行向量数据库更新（Atomic Update）
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        # P5、记录历史审计日志（Audit Trail）
        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    # 删除记忆（执行层）
    def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload.get("data", "")
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        return memory_id

    # 重置记忆（执行层）
    def reset(self):
        """
        重要程度相当于 rm -rf * 了
        一键销毁所有记忆数据、清空历史记录，并重新初始化整个存储系统，让系统回到刚安装时的“出厂设置”状态。
        """
        logger.warning("重置所有记忆")

        # P1、如果数据库连接存在，直接删除 history 表
        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()

        # P2、重新创建一个新的 SQLiteManager 实例
        self.db = SQLiteManager(self.config.history_db_path)

        # P3、摧毁并重建向量数据库
        if hasattr(self.vector_store, "reset"):
            # 情况 A: 如果当前的向量存储类自己实现了 reset 方法
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            # 情况 B: 如果没有原生 reset 方法，则手动执行“删除 + 重建”
            logger.warning("Vector store does not support reset. Skipping.")
            # 1. 删除集合 (Collection) -> 数据全丢
            self.vector_store.delete_col()
            # 2. 重新创建集合 -> 回到空状态
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )

        # 用于捕获事件（可忽略）
        capture_event("mem0.reset", self, {"sync_type": "sync"})


# 异步记忆
class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        pass
