from __future__ import annotations

import json
import os
import re
import urllib.request

from src.extraction.base import BaseExtractor
from src.io.schema import SpanPrediction


TYPE_MAP = {
    "TRIỆU_CHỨNG": "symptom",
    "TÊN_XÉT_NGHIỆM": "lab_test",
    "KẾT_QUẢ_XÉT_NGHIỆM": "lab_result",
    "CHẨN_ĐOÁN": "diagnosis",
    "THUỐC": "drug",
}
VALID_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}

SYSTEM_PROMPT = """Bạn là hệ thống trích xuất thông tin y khoa tiếng Việt.
Chỉ trả về JSON, không giải thích và không dùng markdown.
Phát hiện đầy đủ các cụm từ y tế thuộc đúng năm loại:
TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC.
Mỗi entity phải trích nguyên văn liên tục từ văn bản. Không đưa tuổi, tên, địa chỉ,
thời gian, thủ thuật điều trị hoặc câu mô tả chung thành entity.
Assertion chỉ áp dụng cho TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC và chỉ gồm:
isNegated, isFamily, isHistorical. Không tự sinh mã ICD hoặc RxNorm.
Schema: {"entities":[{"text":"...","position":[start,end],"type":"...","assertions":[]}]}
position là offset ký tự zero-based, end-exclusive trong đúng văn bản đầu vào."""


class LlmExtractor(BaseExtractor):
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.available = self.enabled
        self.warned = False
        self.parse_warnings = 0

    def _endpoint(self) -> str:
        base_url = str(self.config.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
        return f"{base_url}/chat/completions"

    def _request(self, raw_text: str) -> str:
        model = self.config.get("model", "Qwen/Qwen3-8B")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            "temperature": float(self.config.get("temperature", 0.0)),
            "max_tokens": int(self.config.get("max_tokens", 4096)),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.config.get("response_format_json", True):
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(str(self.config.get("api_key_env", "LLM_API_KEY")), "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(self.config.get("timeout_seconds", 180))) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])

    @staticmethod
    def _parse_json(content: str) -> list[dict]:
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return LlmExtractor._recover_entities(cleaned)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        entities = payload.get("entities", []) if isinstance(payload, dict) else []
        if isinstance(entities, dict):
            entities = [entities]
        return [item for item in entities if isinstance(item, dict)]

    @staticmethod
    def _recover_entities(content: str) -> list[dict]:
        entities_match = re.search(r'"entities"\s*:\s*\[', content)
        array_start = entities_match.end() if entities_match else content.find("[") + 1
        if array_start <= 0:
            return []
        recovered: list[dict] = []
        object_start: int | None = None
        depth = 0
        in_string = False
        escaped = False
        for index in range(array_start, len(content)):
            char = content[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    object_start = index
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and object_start is not None:
                    try:
                        item = json.loads(content[object_start : index + 1])
                    except json.JSONDecodeError:
                        item = None
                    if isinstance(item, dict):
                        recovered.append(item)
                    object_start = None
        return recovered

    @staticmethod
    def _align(raw_text: str, mention: str, proposed) -> tuple[int, int] | None:
        if isinstance(proposed, list) and len(proposed) == 2:
            try:
                start, end = int(proposed[0]), int(proposed[1])
            except (TypeError, ValueError):
                start, end = -1, -1
            if 0 <= start < end <= len(raw_text) and raw_text[start:end] == mention:
                return start, end
        occurrences = [match.start() for match in re.finditer(re.escape(mention), raw_text, flags=re.IGNORECASE)]
        if not occurrences:
            return None
        if isinstance(proposed, list) and proposed:
            try:
                approximate = int(proposed[0])
                start = min(occurrences, key=lambda value: abs(value - approximate))
            except (TypeError, ValueError):
                start = occurrences[0]
        else:
            start = occurrences[0]
        return start, start + len(mention)

    @staticmethod
    def _normalize_assertions(assertions) -> list[str]:
        if assertions is None:
            return []
        if not isinstance(assertions, list):
            assertions = [assertions]

        normalized: list[str] = []
        seen: set[str] = set()
        for value in assertions:
            candidate = ""
            if isinstance(value, str):
                candidate = value.strip()
            elif isinstance(value, dict):
                for key in ("value", "name", "type", "label", "assertion"):
                    nested_value = value.get(key)
                    if isinstance(nested_value, str):
                        candidate = nested_value.strip()
                        break
                if not candidate:
                    for label in VALID_ASSERTIONS:
                        if value.get(label) is True and label not in seen:
                            normalized.append(label)
                            seen.add(label)
            elif value is not None:
                candidate = str(value).strip()

            if candidate in VALID_ASSERTIONS and candidate not in seen:
                normalized.append(candidate)
                seen.add(candidate)
        return normalized

    def extract(self, record, sections) -> list[SpanPrediction]:
        if not self.available:
            return []
        try:
            content = self._request(record.raw_text)
        except Exception as exc:
            self.available = False
            if not self.warned:
                print(f"Warning: LLM extractor disabled after server error: {exc}")
                self.warned = True
            return []
        try:
            items = self._parse_json(content)
        except Exception as exc:
            items = []
            if self.parse_warnings < 5:
                print(f"Warning: skipped malformed LLM JSON for record {record.record_id}: {exc}")
                self.parse_warnings += 1
        if not items and content.strip() and self.parse_warnings < 5:
            print(f"Warning: no recoverable LLM entities for record {record.record_id}; continuing with local sources")
            self.parse_warnings += 1
        spans: list[SpanPrediction] = []
        seen: set[tuple[int, int, str]] = set()
        for item in items:
            mention = str(item.get("text", "")).strip()
            concept_type = TYPE_MAP.get(str(item.get("type", "")).upper())
            if not mention or concept_type is None:
                continue
            aligned = self._align(record.raw_text, mention, item.get("position"))
            if aligned is None:
                continue
            start, end = aligned
            key = start, end, concept_type
            if key in seen:
                continue
            seen.add(key)
            assertions = self._normalize_assertions(item.get("assertions", []))
            if concept_type not in {"symptom", "diagnosis", "drug"}:
                assertions = []
            spans.append(
                SpanPrediction(
                    text=record.raw_text[start:end],
                    start=start,
                    end=end,
                    concept_type=concept_type,
                    assertion=assertions[0] if assertions else "isPresent",
                    assertions=assertions,
                    confidence=float(self.config.get("confidence", 0.93)),
                    source="llm_qwen3",
                )
            )
        return sorted(spans, key=lambda span: (span.start, span.end))
