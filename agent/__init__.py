"""Agent 核心 — 构建与运行，三步链路：识别 → 精确查找 → 语义检索"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import load_config
from database import ShipDatabase
from tools import build_tools

logger = logging.getLogger(__name__)

# ── Skills 加载 ────────────────────────────────

def _load_skills(skills_dir: str | Path = "skills") -> str:
    """从 skills/ 目录加载所有 .md 文件，拼接为技能文本。"""
    skills_path = Path(skills_dir)
    if not skills_path.is_dir():
        logger.warning("skills 目录不存在: %s", skills_path)
        return ""

    parts = []
    for md_file in sorted(skills_path.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                parts.append(content)
        except Exception as e:
            logger.warning("加载 skill 失败 %s: %s", md_file, e)

    return "\n\n---\n\n".join(parts)


# ── System Prompt ──────────────────────────────

_BASE_PROMPT = """你是船弦号识别助手。根据VLM预识别结果决策操作。

工具：lookup_by_hull_number(弦号精确查找)、retrieve_by_description(描述语义检索)

执行流程（严格按顺序）：
1. 无弦号 → 直接 retrieve_by_description
2. 有弦号 → lookup_by_hull_number
   - found=true → 返回结果
   - found=false → retrieve_by_description 兜底

返回格式：弦号：{hull_number}，描述：{description}，匹配类型：{exact/semantic/none}

规则：不编造弦号/描述，不跳步，不同时调用多个工具"""


def _build_system_prompt(skills_dir: str | Path = "skills") -> str:
    """构建完整 SYSTEM_PROMPT = 基础提示 + skills 技能文件。"""
    skills_text = _load_skills(skills_dir)
    if skills_text:
        return f"{_BASE_PROMPT}\n\n## 参考技能\n\n{skills_text}"
    return _BASE_PROMPT

# ── 无 Few-shot 示例（避免误导 Agent）──


class AgentResult:
    """Agent 运行结果，包含结构化信息供 pipeline 使用。"""

    def __init__(
        self,
        hull_number: str = "",
        description: str = "",
        match_type: str = "none",
        semantic_match_ids: list[str] | None = None,
        answer: str = "",
        hull_box: list[float] | None = None,
        clarity: str = "",
    ):
        self.hull_number = hull_number
        self.description = description
        self.match_type = match_type        # "exact" | "semantic" | "none"
        self.semantic_match_ids = semantic_match_ids or []
        self.answer = answer
        self.hull_box = hull_box            # 弦号框相对坐标 [rx1, ry1, rx2, ry2]
        self.clarity = clarity              # "clear" | "blurry" | ""


class ShipHullAgent:
    """船弦号识别 Agent 封装。链路：VLM预识别 → Agent决策（lookup/retrieve）。"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

        app_cfg = self.config.get("app", {})
        llm_cfg = self.config.get("llm", {})

        logging.basicConfig(
            level=getattr(logging, app_cfg.get("log_level", "INFO").upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        self.db = ShipDatabase(config=self.config)
        # Agent 模式：不含 recognize_ship（VLM 由 pipeline 预调用），含 lookup + retrieve
        self.tools = build_tools(self.db, include_recognize=False)

        # 动态加载 skills 目录，构建完整 system prompt
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
        system_prompt = _build_system_prompt(skills_dir)

        self._llm = ChatOpenAI(
            model=llm_cfg.get("model", "Qwen/Qwen3-VL-4B-AWQ"),
            api_key=llm_cfg.get("api_key", "abc123"),
            base_url=llm_cfg.get("base_url", "http://localhost:7890/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 2048),
        )

        self._agent = create_react_agent(
            model=self._llm,
            tools=self.tools,
            prompt=system_prompt,
        )

    def run(self, query: str) -> str:
        """运行 Agent，返回自然语言回答。"""
        logger.info("收到查询: %s", query[:100])
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            answer = result["messages"][-1].content
            logger.info("回答: %s", answer[:100])
            return answer
        except Exception as e:
            logger.exception("Agent 执行失败")
            return f"查询执行失败: {e}"

    def run_with_result(self, query: str) -> AgentResult:
        """运行 Agent，返回结构化结果（供 pipeline 使用）。"""
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            return self._parse_result(result)
        except Exception as e:
            logger.exception("Agent 执行失败")
            return AgentResult(answer=f"Agent 执行失败: {e}")

    @staticmethod
    def _parse_result(result: dict) -> AgentResult:
        """从 Agent 消息历史中提取结构化结果。"""
        msgs = result.get("messages", [])
        hull_number = ""
        description = ""
        match_type = "none"
        semantic_match_ids: list[str] = []
        answer = msgs[-1].content if msgs else ""
        hull_box = None
        clarity = ""

        for msg in msgs:
            if not isinstance(msg, ToolMessage):
                continue
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue

            # recognize_ship 结果（包含 hull_number, description, clarity, hull_box）
            if "hull_number" in data and "description" in data and "found" not in data and "results" not in data:
                hull_number = data.get("hull_number", "")
                description = data.get("description", "")
                clarity = data.get("clarity", "")
                raw_box = data.get("hull_box")
                if isinstance(raw_box, list) and len(raw_box) == 4:
                    try:
                        coords = [float(v) for v in raw_box]
                        if all(0.0 <= c <= 1.0 for c in coords):
                            hull_box = coords
                    except (ValueError, TypeError):
                        pass

            # lookup_by_hull_number（无论 found 与否都保留 hull_number）
            if "found" in data:
                if data.get("found") is True:
                    match_type = "exact"
                    description = data.get("description", description)
                hull_number = data.get("hull_number", hull_number) or hull_number

            # retrieve_by_description 语义匹配
            if "results" in data:
                results = data["results"]
                if results:
                    semantic_match_ids = [
                        r.get("hull_number", "") for r in results if r.get("hull_number")
                    ]
                    if match_type != "exact":
                        match_type = "semantic"

        return AgentResult(
            hull_number=hull_number,
            description=description,
            match_type=match_type,
            semantic_match_ids=semantic_match_ids,
            answer=answer,
            hull_box=hull_box,
            clarity=clarity,
        )

    def run_verbose(self, query: str) -> list[dict]:
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            trace = []
            for msg in result["messages"]:
                entry = {"type": msg.type, "content": msg.content}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, "tool_call_id"):
                    entry["tool_call_id"] = msg.tool_call_id
                trace.append(entry)
            return trace
        except Exception as e:
            logger.exception("Agent 执行失败")
            return [{"type": "error", "content": f"查询执行失败: {e}"}]


_agent_instance: ShipHullAgent | None = None
_agent_config_hash: int = 0
_agent_lock = threading.Lock()


def create_agent(config: dict[str, Any] | None = None) -> ShipHullAgent:
    global _agent_instance, _agent_config_hash
    config_hash = hash(str(config)) if config is not None else 0
    with _agent_lock:
        if _agent_instance is None or config_hash != _agent_config_hash:
            _agent_instance = ShipHullAgent(config)
            _agent_config_hash = config_hash
    return _agent_instance
