"""
OpenClaw Browser Agent V20 - 6-Stage Pipeline Architecture
===========================================================
Stage 1: ScreenSimplifier   — Extract question blocks, options, instructions,
                               required status, inputs, next button candidates
Stage 2: StructureValidator  — Classify question type (single/multi/text/matrix/
                                slider/date/login/terminal), check missing conditions
Stage 3: QuestionDecider     — AI decides ONLY "what to answer for this question
                                block" (not full page navigation)
Stage 4: RuleBasedExecutor   — Rule-based clicking: label → container → input,
                                with JS event-sequence fallback for radio/checkbox
Stage 5: ActionVerifier      — Verify selection reflected via JS element state,
                                DOM snapshot change, error messages, next button
                                activation, new question appearance
Stage 6: NextButtonHandler   — Click next ONLY after verification passes (or after
                                forced-pass on repeated retries)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from playwright.async_api import (
    BrowserContext,
    Frame,
    Page,
    Playwright,
    async_playwright,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("openclaw_v20")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENCLAW_API_BASE = os.getenv("OPENCLAW_API_BASE", "https://api.openclaw.io/v1")
OPENCLAW_API_KEY = os.getenv("OPENCLAW_API_KEY", "")
AGENT_MODEL = os.getenv("OPENCLAW_MODEL", "gpt-4o")

MAX_STEPS = int(os.getenv("MAX_STEPS", "60"))
MAX_RETRIES_PER_STEP = int(os.getenv("MAX_RETRIES_PER_STEP", "5"))
RETRY_FORCE_THRESHOLD = 2          # after this many successful-but-unverified retry attempts, force-proceed
STEP_DELAY_MS = 600
CLICK_DELAY_MS = 300
NEXT_DELAY_MS = 800

# CSS/attribute selectors that indicate a selected option
SELECTION_CLASS_PATTERNS = [
    "selected", "active", "choice-on", "checked", "on",
    "is-selected", "is-active", "is-checked", "btn-primary",
    "answer-selected", "option-selected",
]

# Elements considered dangerous (logout, submit final, etc.)
DANGER_TEXT_PATTERNS = [
    r"로그아웃", r"탈퇴", r"최종\s*제출", r"설문\s*종료",
    r"logout", r"sign\s*out", r"final\s*submit",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageElement:
    index: int
    tag: str
    text: str
    attrs: dict[str, str] = field(default_factory=dict)
    is_interactive: bool = False
    frame_id: str = "main"
    xpath: str = ""

    @property
    def type_attr(self) -> str:
        return self.attrs.get("type", "").lower()

    @property
    def role(self) -> str:
        return self.attrs.get("role", "").lower()

    @property
    def aria_checked(self) -> str:
        return self.attrs.get("aria-checked", "").lower()

    @property
    def is_radio(self) -> bool:
        return self.type_attr == "radio" or self.role in ("radio",)

    @property
    def is_checkbox(self) -> bool:
        return self.type_attr == "checkbox" or self.role in ("checkbox",)

    @property
    def is_option(self) -> bool:
        return self.is_radio or self.is_checkbox or self.role in ("option",)


@dataclass
class QuestionBlock:
    question_text: str
    question_type: str          # single_select | multi_select | text_input | matrix | slider | date | login | terminal
    options: list[PageElement]
    inputs: list[PageElement]
    next_candidates: list[PageElement]
    is_required: bool = True
    matrix_rows: list[str] = field(default_factory=list)
    matrix_cols: list[str] = field(default_factory=list)
    raw_elements: list[PageElement] = field(default_factory=list)


@dataclass
class Decision:
    action_type: str            # click | type | select | slider | skip
    targets: list[int]          # element indices
    text_value: str = ""
    reasoning: str = ""


@dataclass
class ExecutionResult:
    success: bool
    fallback_used: str = ""     # "" | "js" | "label" | "container"
    navigated: bool = False
    frame_switched: bool = False
    error: str = ""


@dataclass
class VerificationResult:
    selection_reflected: bool
    errors: list[str]
    next_enabled: bool
    all_answered: bool
    should_proceed_to_next: bool
    forced: bool = False
    js_state_confirmed: bool = False
    snapshot_changed: bool = False


@dataclass
class StepContext:
    step: int
    retry: int
    url: str
    question_type: str
    answered: int
    total: int
    same_action_success_count: int = 0
    last_decision_key: str = ""

# ---------------------------------------------------------------------------
# OpenClaw AI client
# ---------------------------------------------------------------------------

class OpenClawClient:
    """Thin async wrapper around the OpenClaw/OpenAI-compatible chat API."""

    def __init__(self, base_url: str = OPENCLAW_API_BASE, api_key: str = OPENCLAW_API_KEY):
        self._base = base_url.rstrip("/")
        self._key = api_key

    async def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        payload = {
            "model": AGENT_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self._base}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# Helper: frame/element extraction
# ---------------------------------------------------------------------------

ELEMENT_EXTRACT_JS = """
() => {
    const elements = [];
    let idx = 0;

    function isVisible(el) {
        const s = window.getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden' || s.opacity === '0') return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
    }

    function getText(el) {
        return (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 200);
    }

    function getAttrs(el) {
        const attrs = {};
        for (const a of el.attributes) attrs[a.name] = a.value;
        return attrs;
    }

    const INTERACTIVE_TAGS = new Set(['input','select','textarea','button','a']);
    const INTERACTIVE_ROLES = new Set([
        'button','link','checkbox','radio','option','switch',
        'menuitem','tab','combobox','listbox','slider','spinbutton'
    ]);

    function walk(node, frameId) {
        if (node.nodeType !== 1) return;
        const tag = node.tagName.toLowerCase();
        const role = (node.getAttribute('role') || '').toLowerCase();
        const isInteractive = INTERACTIVE_TAGS.has(tag) || INTERACTIVE_ROLES.has(role)
            || node.hasAttribute('onclick') || node.tabIndex >= 0;
        const visible = isVisible(node);
        if (!visible && !isInteractive) {
            // still recurse children in case container is display:block but children are visible
        }
        elements.push({
            index: idx++,
            tag: tag,
            text: getText(node),
            attrs: getAttrs(node),
            is_interactive: isInteractive,
            frame_id: frameId,
        });
        for (const child of node.children) walk(child, frameId);
    }

    walk(document.body, 'main');
    return elements;
}
"""

SNAPSHOT_JS = """
() => {
    // Collect a lightweight signature of the current page selection state
    const sig = [];
    // checked inputs
    document.querySelectorAll('input[type=radio]:checked,input[type=checkbox]:checked').forEach(el => {
        sig.push('ck:' + (el.name||'') + ':' + (el.value||el.id||''));
    });
    // aria-checked
    document.querySelectorAll('[aria-checked="true"]').forEach(el => {
        sig.push('ac:' + (el.id || el.className.slice(0,30)));
    });
    // selected class elements
    const SEL = ['selected','active','choice-on','checked','on','is-selected','is-active','is-checked'];
    SEL.forEach(cls => {
        document.querySelectorAll('.' + cls).forEach(el => {
            sig.push(cls + ':' + (el.id || el.getAttribute('data-value') || el.textContent.trim().slice(0,20)));
        });
    });
    return sig.sort().join('|');
}
"""

ELEMENT_STATE_JS = """
(el) => {
    if (!el) return null;
    const classes = Array.from(el.classList);
    const parentCls = el.parentElement ? Array.from(el.parentElement.classList) : [];
    const grandCls = (el.parentElement && el.parentElement.parentElement)
        ? Array.from(el.parentElement.parentElement.classList) : [];
    return {
        checked: el.checked || false,
        aria_checked: el.getAttribute('aria-checked') || '',
        value: el.value || '',
        classes: classes,
        parent_classes: parentCls,
        grand_classes: grandCls,
        selected_ancestor: !!el.closest('.selected,.active,.choice-on,.checked,.on,.is-selected,.is-active,.is-checked'),
    };
}
"""

JS_CLICK_SEQUENCE = """
(el) => {
    if (!el) return false;
    const opts = {bubbles: true, cancelable: true};
    el.dispatchEvent(new MouseEvent('mouseover', opts));
    el.dispatchEvent(new MouseEvent('mousedown', opts));
    el.dispatchEvent(new MouseEvent('mouseup', opts));
    el.click();
    el.dispatchEvent(new MouseEvent('click', opts));
    el.dispatchEvent(new Event('change', opts));
    el.dispatchEvent(new Event('input', opts));
    return true;
}
"""

# ---------------------------------------------------------------------------
# Stage 1 — ScreenSimplifier
# ---------------------------------------------------------------------------

class ScreenSimplifier:
    """Extract question blocks, options, instructions, inputs, next button candidates."""

    async def run(self, page: Page, frames: list[Frame]) -> list[PageElement]:
        all_elements: list[PageElement] = []
        idx_offset = 0
        for frame in frames:
            try:
                raw = await frame.evaluate(ELEMENT_EXTRACT_JS)
                for r in raw:
                    el = PageElement(
                        index=r["index"] + idx_offset,
                        tag=r["tag"],
                        text=r["text"],
                        attrs=r["attrs"],
                        is_interactive=r["is_interactive"],
                        frame_id=r.get("frame_id", "main"),
                    )
                    all_elements.append(el)
                idx_offset += len(raw)
            except Exception as exc:
                log.warning("ScreenSimplifier frame error: %s", exc)
        return all_elements

# ---------------------------------------------------------------------------
# Stage 2 — StructureValidator
# ---------------------------------------------------------------------------

QUESTION_TYPE_KEYWORDS = {
    "single_select": [
        r"하나만\s*선택", r"단일\s*선택", r"다음\s*중\s*하나", r"택1",
    ],
    "multi_select": [
        r"모두\s*선택", r"복수\s*선택", r"해당하는\s*것\s*모두", r"택\d+",
    ],
    "text_input": [
        r"서술", r"작성", r"입력", r"기재", r"써\s*주세요",
    ],
    "matrix": [
        r"각\s*항목", r"해당\s*란에", r"행렬", r"매트릭스",
    ],
    "slider": [
        r"점수", r"만족도", r"얼마나", r"슬라이더",
    ],
    "date": [
        r"날짜", r"언제", r"연도", r"월\s*일",
    ],
    "login": [
        r"아이디", r"비밀번호", r"로그인", r"회원\s*번호",
    ],
}

PANELNOW_PATTERNS = {
    "cover": [r"패널나우", r"panelnow", r"설문\s*참여"],
    "screenout": [r"스크린\s*아웃", r"screenout", r"참여\s*조건에\s*맞지"],
    "complete": [r"설문\s*완료", r"참여\s*감사", r"포인트\s*적립", r"완료\s*되었습니다"],
}


class StructureValidator:
    """Classify question type and check for special page conditions."""

    def classify(self, elements: list[PageElement]) -> QuestionBlock:
        page_text = " ".join(el.text for el in elements if el.text).lower()

        # Check for special page types
        for ptype, patterns in PANELNOW_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, page_text, re.IGNORECASE):
                    return QuestionBlock(
                        question_text=page_text[:200],
                        question_type=ptype,
                        options=[],
                        inputs=[],
                        next_candidates=[],
                    )

        # Find radio/checkbox options
        options = [el for el in elements if el.is_option and el.is_interactive]
        # If no native radio/checkbox, look for clickable option-like elements
        if not options:
            options = [
                el for el in elements
                if el.is_interactive and el.role in ("option", "radio", "checkbox", "button")
                and any(
                    cls in (el.attrs.get("class", ""))
                    for cls in ["option", "choice", "answer", "btn"]
                )
            ]

        # Find text inputs
        text_inputs = [
            el for el in elements
            if el.tag in ("input", "textarea")
            and el.type_attr not in ("radio", "checkbox", "submit", "button", "hidden", "file")
            and el.is_interactive
        ]

        # Find next button candidates
        next_candidates = _find_next_candidates(elements)

        # Detect question type
        qtype = _detect_question_type(elements, options, text_inputs, page_text)

        # Extract question text (first large text block near top)
        question_text = _extract_question_text(elements)

        # Matrix rows/cols
        matrix_rows, matrix_cols = [], []
        if qtype == "matrix":
            matrix_rows, matrix_cols = _extract_matrix(elements)

        is_required = _is_required(page_text)

        return QuestionBlock(
            question_text=question_text,
            question_type=qtype,
            options=options,
            inputs=text_inputs,
            next_candidates=next_candidates,
            is_required=is_required,
            matrix_rows=matrix_rows,
            matrix_cols=matrix_cols,
            raw_elements=elements,
        )


def _find_next_candidates(elements: list[PageElement]) -> list[PageElement]:
    next_kw = re.compile(r"다음|next|계속|진행|확인|submit|완료", re.IGNORECASE)
    candidates = []
    for el in elements:
        if not el.is_interactive:
            continue
        if el.tag in ("button", "a", "input") or el.role == "button":
            if next_kw.search(el.text) or next_kw.search(el.attrs.get("value", "")):
                candidates.append(el)
    return candidates


def _detect_question_type(
    elements: list[PageElement],
    options: list[PageElement],
    text_inputs: list[PageElement],
    page_text: str,
) -> str:
    # Check keyword patterns first
    for qtype, patterns in QUESTION_TYPE_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, page_text, re.IGNORECASE):
                return qtype

    if text_inputs and not options:
        return "text_input"

    if options:
        # Check for matrix (multiple rows of radio groups)
        radio_names = set(
            el.attrs.get("name", "") for el in options
            if el.is_radio and el.attrs.get("name")
        )
        if len(radio_names) > 1:
            return "matrix"
        # Count how many options have checkbox type
        checkboxes = [el for el in options if el.is_checkbox]
        radios = [el for el in options if el.is_radio]
        if checkboxes and not radios:
            return "multi_select"
        return "single_select"

    # Slider
    sliders = [el for el in elements if el.tag == "input" and el.type_attr == "range"]
    if sliders:
        return "slider"

    # Date
    date_inputs = [
        el for el in elements
        if el.tag == "input" and el.type_attr in ("date", "month", "week")
    ]
    if date_inputs:
        return "date"

    # Login
    pw_inputs = [
        el for el in elements
        if el.tag == "input" and el.type_attr == "password"
    ]
    if pw_inputs:
        return "login"

    return "terminal"


def _extract_question_text(elements: list[PageElement]) -> str:
    heading_tags = {"h1", "h2", "h3", "h4", "legend", "label"}
    for el in elements:
        if el.tag in heading_tags and len(el.text) > 5:
            return el.text
    # Fallback: longest text element
    texts = sorted(
        (el for el in elements if len(el.text) > 10),
        key=lambda e: -len(e.text),
    )
    return texts[0].text if texts else ""


def _extract_matrix(elements: list[PageElement]) -> tuple[list[str], list[str]]:
    rows: list[str] = []
    cols: list[str] = []
    th_elements = [el for el in elements if el.tag == "th" and el.text]
    td_elements = [el for el in elements if el.tag == "td" and el.text]
    cols = [el.text for el in th_elements[:10]]
    rows = [el.text for el in td_elements[:20] if len(el.text) < 50]
    return rows, cols


def _is_required(page_text: str) -> bool:
    return bool(re.search(r"필수|required|\*", page_text, re.IGNORECASE))

# ---------------------------------------------------------------------------
# Stage 3 — QuestionDecider (ONLY AI call in pipeline)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """당신은 온라인 설문조사를 자동으로 응답하는 AI 에이전트입니다.
주어진 질문 블록에 대해 선택 또는 입력할 내용을 결정하세요.

규칙:
1. 단일선택(single_select): 가장 일반적이고 중립적인 보기 1개를 선택하세요.
2. 복수선택(multi_select): 적절한 보기 1~3개를 선택하세요.
3. 서술형(text_input): 자연스러운 한국어 답변을 작성하세요.
4. 매트릭스(matrix): 각 행마다 하나씩 선택하세요.
5. 슬라이더(slider): 중간값을 선택하세요.
6. 날짜(date): 합리적인 날짜를 입력하세요.
7. 로그인(login): 제공된 자격증명을 입력하세요.
8. 패널나우 커버 화면(cover): 참여 버튼을 클릭하세요.
9. 스크린아웃(screenout)/완료(complete/terminal): skip 액션을 반환하세요.

응답 형식 (JSON만 반환):
{
  "action_type": "click|type|slider|skip",
  "targets": [<element index>, ...],
  "text_value": "<입력 텍스트, 해당 시>",
  "reasoning": "<간단한 한국어 이유>"
}"""


class QuestionDecider:
    """Stage 3: AI call to decide what action to take for this question block."""

    def __init__(self, client: OpenClawClient):
        self._client = client

    async def decide(
        self,
        block: QuestionBlock,
        context: StepContext,
        credentials: Optional[dict] = None,
    ) -> Decision:
        if block.question_type in ("screenout", "complete", "terminal"):
            return Decision(action_type="skip", targets=[], reasoning="설문 종료/완료 화면")

        # Build user prompt
        options_text = "\n".join(
            f"  [{el.index}] {el.text or el.attrs.get('value', '?')}"
            for el in block.options[:30]
        )
        inputs_text = "\n".join(
            f"  [{el.index}] {el.tag}[type={el.type_attr}] placeholder={el.attrs.get('placeholder','')}"
            for el in block.inputs[:10]
        )
        next_text = "\n".join(
            f"  [{el.index}] {el.text}" for el in block.next_candidates[:5]
        )

        cred_hint = ""
        if credentials and block.question_type == "login":
            cred_hint = (
                f"\n로그인 자격증명: id={credentials.get('id','')}, pw={credentials.get('pw','')}"
            )

        user_msg = (
            f"질문 유형: {block.question_type}\n"
            f"질문: {block.question_text[:300]}\n\n"
            f"보기/옵션:\n{options_text or '없음'}\n\n"
            f"입력 필드:\n{inputs_text or '없음'}\n\n"
            f"다음 버튼 후보:\n{next_text or '없음'}\n\n"
            f"현재 단계: step {context.step} retry {context.retry}{cred_hint}"
        )

        try:
            raw = await self._client.chat(SYSTEM_PROMPT, user_msg)
            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if not json_match:
                raise ValueError(f"No JSON in response: {raw[:100]}")
            data = json.loads(json_match.group())
            return Decision(
                action_type=data.get("action_type", "skip"),
                targets=[int(t) for t in data.get("targets", [])],
                text_value=data.get("text_value", ""),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as exc:
            log.error("QuestionDecider error: %s", exc)
            # Fallback: click first option
            if block.options:
                return Decision(
                    action_type="click",
                    targets=[block.options[0].index],
                    reasoning=f"AI 오류 fallback: {exc}",
                )
            return Decision(action_type="skip", targets=[], reasoning=f"AI 오류: {exc}")

# ---------------------------------------------------------------------------
# Stage 4 — RuleBasedExecutor
# ---------------------------------------------------------------------------

class RuleBasedExecutor:
    """Stage 4: Rule-based click execution (no AI). label→container→input order.
    For radio/checkbox: dispatches full mousedown→mouseup→click→change→input sequence.
    """

    def __init__(self, page: Page, frames: list[Frame]):
        self._page = page
        self._frames = frames

    async def execute(
        self,
        decision: Decision,
        block: QuestionBlock,
    ) -> ExecutionResult:
        if decision.action_type == "skip":
            return ExecutionResult(success=True, fallback_used="skip")

        if decision.action_type == "type":
            return await self._execute_type(decision, block)

        if decision.action_type == "slider":
            return await self._execute_slider(decision, block)

        # Default: click
        return await self._execute_click(decision, block)

    # ------------------------------------------------------------------
    # Click execution
    # ------------------------------------------------------------------

    async def _execute_click(self, decision: Decision, block: QuestionBlock) -> ExecutionResult:
        for target_idx in decision.targets:
            result = await self._click_by_index(target_idx, block)
            if not result.success:
                return result
            await self._page.wait_for_timeout(CLICK_DELAY_MS)
        return ExecutionResult(success=True)

    async def _click_by_index(self, idx: int, block: QuestionBlock) -> ExecutionResult:
        el = _find_element(idx, block)
        if el is None:
            log.warning("Element index %d not found in block", idx)
            return ExecutionResult(success=False, error=f"index {idx} not found")

        # Check if action would be dangerous
        if _is_danger(el):
            log.warning("Blocked dangerous action on element %d: %s", idx, el.text)
            return ExecutionResult(success=False, error="danger action blocked")

        # For radio/checkbox or option-role elements, use JS event sequence
        if el.is_option:
            success = await self._js_click_sequence(idx)
            if success:
                return ExecutionResult(success=True, fallback_used="js")

        # Try native Playwright click strategies: label → container → input
        for strategy in ("native", "label", "container"):
            try:
                nav = await self._playwright_click(idx, strategy)
                return ExecutionResult(success=True, fallback_used=strategy, navigated=nav)
            except Exception as exc:
                log.debug("Click strategy %s failed for %d: %s", strategy, idx, exc)
                continue

        # Final fallback: JS click sequence
        success = await self._js_click_sequence(idx)
        return ExecutionResult(success=success, fallback_used="js", error="" if success else "all strategies failed")

    async def _playwright_click(self, idx: int, strategy: str) -> bool:
        """Returns True if navigation occurred."""
        selector = f"[data-oc-index='{idx}']"
        # We inject data-oc-index attributes before clicking; try nth approach otherwise
        for frame in self._frames:
            try:
                locator = frame.locator(f":nth-match(*[data-oc-index='{idx}'], 1)")
                pre_url = self._page.url
                await locator.click(timeout=3000)
                await self._page.wait_for_timeout(200)
                return self._page.url != pre_url
            except Exception:
                pass
        raise RuntimeError(f"playwright click strategy {strategy} failed for {idx}")

    async def _js_click_sequence(self, idx: int) -> bool:
        """Dispatch full mouse event sequence via JS. Works for most survey frameworks."""
        js = f"""
        () => {{
            const el = document.querySelector('[data-oc-index="{idx}"]');
            if (!el) return false;
            const opts = {{bubbles: true, cancelable: true}};
            el.dispatchEvent(new MouseEvent('mouseover', opts));
            el.dispatchEvent(new MouseEvent('mousedown', opts));
            el.dispatchEvent(new MouseEvent('mouseup', opts));
            el.click();
            el.dispatchEvent(new MouseEvent('click', opts));
            el.dispatchEvent(new Event('change', opts));
            el.dispatchEvent(new Event('input', opts));
            return true;
        }}
        """
        for frame in self._frames:
            try:
                result = await frame.evaluate(js)
                if result:
                    return True
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # Type execution
    # ------------------------------------------------------------------

    async def _execute_type(self, decision: Decision, block: QuestionBlock) -> ExecutionResult:
        targets = decision.targets or [el.index for el in block.inputs[:1]]
        for idx in targets:
            el = _find_element(idx, block)
            if el is None:
                continue
            for frame in self._frames:
                try:
                    selector = f'[data-oc-index="{idx}"]'
                    await frame.fill(selector, decision.text_value, timeout=3000)
                    return ExecutionResult(success=True)
                except Exception:
                    pass
        return ExecutionResult(success=False, error="type failed")

    # ------------------------------------------------------------------
    # Slider execution
    # ------------------------------------------------------------------

    async def _execute_slider(self, decision: Decision, block: QuestionBlock) -> ExecutionResult:
        sliders = [el for el in block.raw_elements if el.tag == "input" and el.type_attr == "range"]
        if not sliders:
            return ExecutionResult(success=False, error="no slider found")
        el = sliders[0]
        min_val = float(el.attrs.get("min", "0"))
        max_val = float(el.attrs.get("max", "100"))
        mid_val = str(int((min_val + max_val) / 2))
        for frame in self._frames:
            try:
                selector = f'[data-oc-index="{el.index}"]'
                await frame.fill(selector, mid_val, timeout=3000)
                await frame.dispatch_event(selector, "change")
                return ExecutionResult(success=True)
            except Exception:
                pass
        return ExecutionResult(success=False, error="slider fill failed")


def _find_element(idx: int, block: QuestionBlock) -> Optional[PageElement]:
    for el in block.raw_elements:
        if el.index == idx:
            return el
    return None


def _is_danger(el: PageElement) -> bool:
    for pat in DANGER_TEXT_PATTERNS:
        if re.search(pat, el.text, re.IGNORECASE):
            return True
    return False

# ---------------------------------------------------------------------------
# Stage 5 — ActionVerifier
# ---------------------------------------------------------------------------

class ActionVerifier:
    """Stage 5: Verify that the action was reflected in the DOM.

    Fixes for the V20 bug where radio/option clicks succeed but verification
    fails due to CSS-class based selection state (not native .checked):

    1. _check_element_state_via_js() — direct JS query on clicked element
    2. _snapshot_changed() — compare DOM signature before/after ANY change
    3. Retry-force logic — after RETRY_FORCE_THRESHOLD successful unverified
       retries, force should_proceed_to_next=True when next_enabled=True
    4. Single-select auto-advance — if next_enabled after option click, proceed
    """

    def __init__(self, page: Page, frames: list[Frame]):
        self._page = page
        self._frames = frames

    async def verify(
        self,
        decision: Decision,
        execution: ExecutionResult,
        block: QuestionBlock,
        context: StepContext,
        pre_snapshot: str,
    ) -> VerificationResult:

        if decision.action_type == "skip":
            return VerificationResult(
                selection_reflected=True,
                errors=[],
                next_enabled=False,
                all_answered=True,
                should_proceed_to_next=False,
            )

        await self._page.wait_for_timeout(400)

        # Collect post-action state
        errors = await self._collect_errors()
        next_enabled = await self._is_next_enabled(block)
        post_snapshot = await self._take_snapshot()
        snapshot_changed = _snapshot_changed(pre_snapshot, post_snapshot)

        # Check selection state via JS on clicked element(s)
        js_state_confirmed = False
        if execution.success and decision.action_type == "click" and decision.targets:
            js_state_confirmed = await self._check_element_state_via_js(decision.targets[0])

        # Determine if selection was reflected
        selection_reflected = (
            js_state_confirmed
            or snapshot_changed
            or execution.navigated
        )

        # Count answered questions (heuristic)
        all_answered, answered, total = await self._count_answered(block)

        # Retry-force logic:
        # If the same click succeeded N times but verification never confirmed it,
        # and next is available → force-proceed
        forced = False
        if (
            not selection_reflected
            and execution.success
            and next_enabled
            and context.same_action_success_count >= RETRY_FORCE_THRESHOLD
        ):
            log.info(
                "Force-proceed: same action succeeded %d times with next_enabled=True",
                context.same_action_success_count,
            )
            selection_reflected = True
            forced = True

        # Single-select auto-advance:
        # After clicking an option, if next is enabled, always try proceeding
        # (selection may not be detectable via DOM on some frameworks)
        auto_advance = (
            block.question_type == "single_select"
            and decision.action_type == "click"
            and execution.success
            and next_enabled
            and not execution.navigated
        )

        should_proceed = (
            (selection_reflected and next_enabled and (all_answered or auto_advance or forced))
            or execution.navigated
            or auto_advance
            or forced
        )

        log.info(
            "Verification: reflected=%s js=%s snapshot_changed=%s forced=%s "
            "errors=%s next=%s answered=%d/%d should_next=%s",
            selection_reflected, js_state_confirmed, snapshot_changed, forced,
            errors, next_enabled, answered, total, should_proceed,
        )

        return VerificationResult(
            selection_reflected=selection_reflected,
            errors=errors,
            next_enabled=next_enabled,
            all_answered=all_answered,
            should_proceed_to_next=should_proceed,
            forced=forced,
            js_state_confirmed=js_state_confirmed,
            snapshot_changed=snapshot_changed,
        )

    # ------------------------------------------------------------------
    # JS-based element state check (fix for radio/CSS-selection bug)
    # ------------------------------------------------------------------

    async def _check_element_state_via_js(self, target_idx: int) -> bool:
        """Check if the clicked element shows selected state via any mechanism."""
        js = f"""
        () => {{
            const el = document.querySelector('[data-oc-index="{target_idx}"]');
            if (!el) return null;
            const SELECTION_CLASSES = [
                'selected','active','choice-on','checked','on',
                'is-selected','is-active','is-checked','btn-primary',
                'answer-selected','option-selected'
            ];
            const classes = Array.from(el.classList);
            const parentCls = el.parentElement ? Array.from(el.parentElement.classList) : [];
            const grandCls = (el.parentElement && el.parentElement.parentElement)
                ? Array.from(el.parentElement.parentElement.classList) : [];
            const allClasses = [...classes, ...parentCls, ...grandCls];
            const hasSelClass = allClasses.some(c => SELECTION_CLASSES.includes(c));
            const ariaChecked = el.getAttribute('aria-checked');
            const nativeChecked = el.checked || false;
            const ancestor = el.closest(SELECTION_CLASSES.map(c => '.' + c).join(',')) !== null;
            return {{
                native_checked: nativeChecked,
                aria_checked: ariaChecked,
                has_selection_class: hasSelClass,
                selected_ancestor: ancestor,
            }};
        }}
        """
        for frame in self._frames:
            try:
                state = await frame.evaluate(js)
                if state is None:
                    continue
                confirmed = (
                    state.get("native_checked")
                    or state.get("aria_checked") == "true"
                    or state.get("has_selection_class")
                    or state.get("selected_ancestor")
                )
                log.debug("JS element state for idx %d: %s → confirmed=%s", target_idx, state, confirmed)
                return bool(confirmed)
            except Exception as exc:
                log.debug("_check_element_state_via_js error: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Snapshot comparison
    # ------------------------------------------------------------------

    async def _take_snapshot(self) -> str:
        for frame in self._frames:
            try:
                sig = await frame.evaluate(SNAPSHOT_JS)
                return sig
            except Exception:
                pass
        return ""

    # ------------------------------------------------------------------
    # Error detection
    # ------------------------------------------------------------------

    async def _collect_errors(self) -> list[str]:
        error_js = """
        () => {
            const ERROR_SELECTORS = [
                '.error','.alert-danger','.validation-error','.error-message',
                '[role="alert"]','.msg-error','.invalid-feedback','.field-error'
            ];
            const msgs = [];
            for (const sel of ERROR_SELECTORS) {
                document.querySelectorAll(sel).forEach(el => {
                    const t = (el.innerText || '').trim();
                    if (t) msgs.push(t);
                });
            }
            return msgs.slice(0, 5);
        }
        """
        for frame in self._frames:
            try:
                return await frame.evaluate(error_js)
            except Exception:
                pass
        return []

    # ------------------------------------------------------------------
    # Next button state
    # ------------------------------------------------------------------

    async def _is_next_enabled(self, block: QuestionBlock) -> bool:
        if not block.next_candidates:
            return False
        next_js = """
        () => {
            const KW = /다음|next|계속|진행|확인|submit|완료/i;
            const candidates = [...document.querySelectorAll('button,a,input[type=button],input[type=submit]')]
                .filter(el => KW.test(el.textContent) || KW.test(el.value || ''));
            if (!candidates.length) return false;
            return candidates.some(el => !el.disabled && !el.classList.contains('disabled'));
        }
        """
        for frame in self._frames:
            try:
                return bool(await frame.evaluate(next_js))
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # Count answered questions
    # ------------------------------------------------------------------

    async def _count_answered(self, block: QuestionBlock) -> tuple[bool, int, int]:
        if block.question_type in ("terminal", "complete", "screenout", "cover"):
            return True, 1, 1

        count_js = """
        () => {
            const total = document.querySelectorAll(
                'input[type=radio]:not([disabled]),input[type=checkbox]:not([disabled]),' +
                '[role=radio],[role=checkbox],[role=option]'
            ).length;
            const checked = document.querySelectorAll(
                'input[type=radio]:checked,input[type=checkbox]:checked,' +
                '[aria-checked="true"]'
            ).length;
            // Also count CSS-selected elements
            const cssSelected = document.querySelectorAll(
                '.selected,.active,.choice-on,.is-selected,.is-active,.is-checked'
            ).length;
            return {total, checked, css_selected: cssSelected};
        }
        """
        for frame in self._frames:
            try:
                data = await frame.evaluate(count_js)
                total = data.get("total", 0)
                checked = max(data.get("checked", 0), data.get("css_selected", 0))
                required = block.is_required
                answered = checked > 0
                all_answered = not required or answered
                return all_answered, checked, total
            except Exception:
                pass
        return False, 0, 0


def _snapshot_changed(before: str, after: str) -> bool:
    """Return True if any selection-state change was detected between snapshots."""
    if before == after:
        return False
    # Ignore whitespace/order differences
    def normalize(s: str) -> set:
        return set(s.split("|")) if s else set()
    return normalize(before) != normalize(after)


async def take_pre_snapshot(frames: list[Frame]) -> str:
    for frame in frames:
        try:
            sig = await frame.evaluate(SNAPSHOT_JS)
            return sig
        except Exception:
            pass
    return ""

# ---------------------------------------------------------------------------
# Stage 6 — NextButtonHandler
# ---------------------------------------------------------------------------

class NextButtonHandler:
    """Stage 6: Click next ONLY when verification passes or forced."""

    def __init__(self, page: Page, frames: list[Frame]):
        self._page = page
        self._frames = frames

    async def handle(
        self,
        verification: VerificationResult,
        block: QuestionBlock,
    ) -> bool:
        """Returns True if next was clicked (and possibly navigated)."""
        if not verification.should_proceed_to_next:
            return False

        if verification.errors:
            log.info("Skipping next click due to validation errors: %s", verification.errors)
            return False

        # Try clicking next button candidates
        for candidate in block.next_candidates:
            clicked = await self._click_next(candidate)
            if clicked:
                await self._page.wait_for_timeout(NEXT_DELAY_MS)
                log.info("Next button clicked: [%d] %s", candidate.index, candidate.text)
                return True

        # Fallback: find and click any enabled next button dynamically
        return await self._click_next_dynamic()

    async def _click_next(self, el: PageElement) -> bool:
        idx = el.index
        js = f"""
        () => {{
            const el = document.querySelector('[data-oc-index="{idx}"]');
            if (!el || el.disabled) return false;
            el.click();
            return true;
        }}
        """
        for frame in self._frames:
            try:
                ok = await frame.evaluate(js)
                if ok:
                    return True
            except Exception:
                pass
        return False

    async def _click_next_dynamic(self) -> bool:
        js = """
        () => {
            const KW = /다음|next|계속|진행|확인|submit|완료/i;
            const btns = [...document.querySelectorAll('button,a,input[type=button],input[type=submit]')]
                .filter(el => (KW.test(el.textContent) || KW.test(el.value||'')) && !el.disabled);
            if (!btns.length) return false;
            btns[0].click();
            return true;
        }
        """
        for frame in self._frames:
            try:
                ok = await frame.evaluate(js)
                if ok:
                    return True
            except Exception:
                pass
        return False

# ---------------------------------------------------------------------------
# Frame & index injection utilities
# ---------------------------------------------------------------------------

async def collect_frames(page: Page) -> list[Frame]:
    """Collect all frames including nested iframes."""
    frames: list[Frame] = [page.main_frame]
    for frame in page.frames:
        if frame != page.main_frame:
            frames.append(frame)
    return frames


async def inject_oc_indices(frames: list[Frame]) -> None:
    """Inject data-oc-index attributes so elements can be selected by JS/Playwright."""
    js = """
    () => {
        let idx = 0;
        document.querySelectorAll('*').forEach(el => {
            el.setAttribute('data-oc-index', String(idx++));
        });
        return idx;
    }
    """
    for frame in frames:
        try:
            count = await frame.evaluate(js)
            log.debug("Injected %d data-oc-index attributes in frame %s", count, frame.url)
        except Exception as exc:
            log.debug("inject_oc_indices frame error: %s", exc)


async def block_auxiliary_requests(context: BrowserContext) -> None:
    """Block analytics, tracking, and other non-essential requests."""
    block_patterns = [
        "google-analytics.com", "googletagmanager.com", "facebook.net",
        "doubleclick.net", "ads.twitter.com", "hotjar.com", "mixpanel.com",
        "segment.io", "cdn.mxpnl.com", "stats.g.doubleclick.net",
    ]

    async def route_handler(route, request):
        url = request.url
        if any(p in url for p in block_patterns):
            await route.abort()
        else:
            await route.continue_()

    await context.route("**/*", route_handler)

# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

class OpenClawPipelineAgent:
    """V20 6-stage pipeline survey automation agent."""

    def __init__(
        self,
        start_url: str,
        credentials: Optional[dict] = None,
        headless: bool = True,
    ):
        self._start_url = start_url
        self._credentials = credentials or {}
        self._headless = headless
        self._client = OpenClawClient()

        # State
        self._multi_select_memory: dict[str, list[int]] = {}  # question_text → selected indices
        self._matrix_row_state: dict[str, str] = {}           # row_text → selected col

    async def run(self) -> dict[str, Any]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._headless)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            await block_auxiliary_requests(context)
            page = await context.new_page()

            try:
                result = await self._run_pipeline(page)
            finally:
                await browser.close()

            return result

    async def _run_pipeline(self, page: Page) -> dict[str, Any]:
        simplifier = ScreenSimplifier()
        validator = StructureValidator()
        decider = QuestionDecider(self._client)

        log.info("Navigating to %s", self._start_url)
        await page.goto(self._start_url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(STEP_DELAY_MS)

        completed_steps = 0
        consecutive_failures = 0
        same_action_success_count = 0
        last_decision_key = ""

        for step in range(MAX_STEPS):
            url = page.url
            log.info("\n=== step %d ===\nurl: %s", step, url)

            # Collect frames (main + iframes)
            frames = await collect_frames(page)

            # Zero-element frame recovery: if main frame is empty, try child frames
            frames = await self._recover_frames_if_needed(frames, page)

            # Inject element indices
            await inject_oc_indices(frames)

            # Stage 1: Extract page elements
            elements = await simplifier.run(page, frames)
            if not elements:
                log.warning("No elements extracted — page may be loading, waiting...")
                await page.wait_for_timeout(2000)
                continue

            # Stage 2: Validate structure & classify question
            block = validator.classify(elements)
            log.info(
                "question_type: %s | options: %d | inputs: %d",
                block.question_type, len(block.options), len(block.inputs),
            )

            # Handle special page types
            if block.question_type == "cover":
                clicked = await self._handle_cover(page, frames, block)
                if clicked:
                    await page.wait_for_timeout(NEXT_DELAY_MS)
                    completed_steps += 1
                    consecutive_failures = 0
                    continue

            if block.question_type in ("complete", "terminal"):
                log.info("Survey complete/terminal page detected — stopping.")
                return {"status": "complete", "steps": completed_steps, "url": url}

            if block.question_type == "screenout":
                log.info("Screenout page detected — stopping.")
                return {"status": "screenout", "steps": completed_steps, "url": url}

            # Login injection
            if block.question_type == "login" and self._credentials:
                success = await self._handle_login(page, frames, block)
                if success:
                    await page.wait_for_timeout(NEXT_DELAY_MS)
                    completed_steps += 1
                    consecutive_failures = 0
                    continue

            # Take pre-action snapshot for Stage 5
            pre_snapshot = await take_pre_snapshot(frames)

            # Context for retry-force logic
            ctx = StepContext(
                step=step,
                retry=0,
                url=url,
                question_type=block.question_type,
                answered=0,
                total=len(block.options),
                same_action_success_count=same_action_success_count,
                last_decision_key=last_decision_key,
            )

            # Stage 3: AI decides what to do
            decision = await decider.decide(block, ctx, self._credentials)
            log.info(
                "Decision: type=%s targets=%s reasoning=%s",
                decision.action_type, decision.targets, decision.reasoning,
            )

            # Multi-select memory: avoid re-selecting already selected options
            if block.question_type == "multi_select":
                decision = self._apply_multi_select_memory(block, decision)

            # Inner retry loop
            retry_succeeded = False
            for retry in range(MAX_RETRIES_PER_STEP):
                ctx.retry = retry

                executor = RuleBasedExecutor(page, frames)
                verifier = ActionVerifier(page, frames)
                next_handler = NextButtonHandler(page, frames)

                # Stage 4: Execute action
                execution = await executor.execute(decision, block)
                log.info(
                    "Execution: success=%s fallback=%s navigated=%s switched=%s",
                    execution.success, execution.fallback_used,
                    execution.navigated, execution.frame_switched,
                )

                if not execution.success:
                    log.warning("Execution failed: %s", execution.error)
                    consecutive_failures += 1
                    break

                # Track same-action success count
                decision_key = f"{decision.action_type}:{decision.targets}"
                if decision_key == last_decision_key:
                    same_action_success_count += 1
                else:
                    same_action_success_count = 1
                    last_decision_key = decision_key

                ctx.same_action_success_count = same_action_success_count

                # If navigated, break out and move to next step
                if execution.navigated:
                    log.info("Navigation detected — advancing to next step")
                    retry_succeeded = True
                    break

                # Stage 5: Verify
                verification = await verifier.verify(
                    decision, execution, block, ctx, pre_snapshot
                )

                # Stage 6: Click next if appropriate
                if verification.should_proceed_to_next:
                    clicked_next = await next_handler.handle(verification, block)
                    if clicked_next or verification.forced or verification.selection_reflected:
                        retry_succeeded = True
                        # Update multi-select memory
                        if block.question_type == "multi_select":
                            self._update_multi_select_memory(block, decision)
                        break
                else:
                    # Not ready to proceed — retry with same decision
                    log.info("Verification not satisfied — retry %d/%d", retry + 1, MAX_RETRIES_PER_STEP)
                    await page.wait_for_timeout(STEP_DELAY_MS)
                    pre_snapshot = await take_pre_snapshot(frames)

            if retry_succeeded:
                completed_steps += 1
                consecutive_failures = 0
                same_action_success_count = 0
                last_decision_key = ""
                await page.wait_for_timeout(STEP_DELAY_MS)
            else:
                consecutive_failures += 1
                log.warning("Step %d failed after all retries", step)
                if consecutive_failures >= 3:
                    log.error("3 consecutive failures — aborting")
                    return {"status": "failed", "steps": completed_steps, "url": page.url}

        return {"status": "max_steps", "steps": completed_steps, "url": page.url}

    # ------------------------------------------------------------------
    # Frame recovery (zero-element frame)
    # ------------------------------------------------------------------

    async def _recover_frames_if_needed(
        self, frames: list[Frame], page: Page
    ) -> list[Frame]:
        """If the main frame has no interactive elements, try child frames."""
        main_count_js = "() => document.querySelectorAll('input,button,a,select,textarea').length"
        try:
            count = await page.evaluate(main_count_js)
            if count == 0 and len(frames) > 1:
                log.info("Main frame empty — using child frames as primary")
                return frames[1:]
        except Exception:
            pass
        return frames

    # ------------------------------------------------------------------
    # Cover page handler
    # ------------------------------------------------------------------

    async def _handle_cover(
        self, page: Page, frames: list[Frame], block: QuestionBlock
    ) -> bool:
        """Click participate/agree button on cover pages."""
        participate_kw = re.compile(
            r"참여|시작|동의|agree|start|begin|참가|입장", re.IGNORECASE
        )
        for el in block.raw_elements:
            if el.is_interactive and participate_kw.search(el.text):
                js = f"""
                () => {{
                    const el = document.querySelector('[data-oc-index="{el.index}"]');
                    if (!el) return false;
                    el.click();
                    return true;
                }}
                """
                for frame in frames:
                    try:
                        ok = await frame.evaluate(js)
                        if ok:
                            log.info("Cover: clicked '%s'", el.text)
                            return True
                    except Exception:
                        pass
        return False

    # ------------------------------------------------------------------
    # Login handler
    # ------------------------------------------------------------------

    async def _handle_login(
        self, page: Page, frames: list[Frame], block: QuestionBlock
    ) -> bool:
        """Inject login credentials into id/password fields."""
        creds = self._credentials
        if not creds:
            return False
        filled = 0
        for inp in block.inputs:
            if inp.type_attr == "password":
                val = creds.get("pw", creds.get("password", ""))
            elif inp.type_attr in ("text", "email", "tel"):
                val = creds.get("id", creds.get("username", ""))
            else:
                continue
            if not val:
                continue
            for frame in frames:
                try:
                    selector = f'[data-oc-index="{inp.index}"]'
                    await frame.fill(selector, val, timeout=3000)
                    filled += 1
                    break
                except Exception:
                    pass

        if filled:
            # Click submit/login button
            submit_kw = re.compile(r"로그인|login|sign\s*in|확인|submit", re.IGNORECASE)
            for el in block.raw_elements:
                if el.is_interactive and submit_kw.search(el.text):
                    js = f"""
                    () => {{
                        const el = document.querySelector('[data-oc-index="{el.index}"]');
                        if (!el) return false;
                        el.click();
                        return true;
                    }}
                    """
                    for frame in frames:
                        try:
                            ok = await frame.evaluate(js)
                            if ok:
                                return True
                        except Exception:
                            pass
        return filled > 0

    # ------------------------------------------------------------------
    # Multi-select memory
    # ------------------------------------------------------------------

    def _apply_multi_select_memory(
        self, block: QuestionBlock, decision: Decision
    ) -> Decision:
        """Avoid re-selecting already selected options."""
        key = block.question_text[:100]
        already = self._multi_select_memory.get(key, [])
        new_targets = [t for t in decision.targets if t not in already]
        if not new_targets and block.options:
            # All targets already selected — pick an un-selected option
            unused = [el.index for el in block.options if el.index not in already]
            new_targets = unused[:1]
        if new_targets != decision.targets:
            log.debug(
                "Multi-select memory: filtered targets %s → %s",
                decision.targets, new_targets,
            )
        return Decision(
            action_type=decision.action_type,
            targets=new_targets,
            text_value=decision.text_value,
            reasoning=decision.reasoning,
        )

    def _update_multi_select_memory(self, block: QuestionBlock, decision: Decision) -> None:
        key = block.question_text[:100]
        existing = self._multi_select_memory.get(key, [])
        self._multi_select_memory[key] = list(set(existing + decision.targets))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenClaw Browser Agent V20 Pipeline")
    parser.add_argument("url", help="Survey start URL")
    parser.add_argument("--id", help="Login ID/username", default="")
    parser.add_argument("--pw", help="Login password", default="")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    credentials = {}
    if args.id:
        credentials["id"] = args.id
    if args.pw:
        credentials["pw"] = args.pw

    agent = OpenClawPipelineAgent(
        start_url=args.url,
        credentials=credentials,
        headless=args.headless,
    )

    result = await agent.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in ("complete",) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
