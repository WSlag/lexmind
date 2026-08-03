"""Intermediate Representation (IR) parser.

LLMs frequently wrap structured output in prose, code fences, or tags, and
sometimes emit malformed JSON (trailing commas, unquoted keys, single-quoted
strings). This module extracts the *intermediate representation* from raw
model output and converts it into a plain Python ``dict`` / ``list`` without
requiring strict JSON compliance — the pattern used by robust contract-review
pipelines (e.g. contract-playbook-ai) to avoid brittle ``json.loads`` failures.

Pipeline stages may also request tagged output (``<ir>...</ir>``). The parser
prefers the tagged block, then fenced blocks, then the first balanced JSON
structure, and finally applies tolerant fixes before attempting to decode.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Fallback when the model returns a Python-style literal (True/None) instead
# of JSON (true/null) or uses single quotes.
_PY_LITERALS = {
    "True": "true",
    "False": "false",
    "None": "null",
}

_TAG_RE = re.compile(r"<ir>(.*?)</ir>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|python|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Keys or values written with single quotes: 'foo': 'bar'.
_SINGLE_QUOTE_KEY_RE = re.compile(r"(?<![\\:\w])\'([^:\'\\]+)\'\s*:", re.MULTILINE)
_SINGLE_QUOTE_VALUE_RE = re.compile(r":\s*\'([^\'\\]*)\'", re.MULTILINE)


def parse_ir(text: str) -> Any:
    """Convert model output into Python objects.

    Tries, in order: an explicit ``<ir>`` block, a fenced code block, the whole
    trimmed output, and finally any balanced ``{...}`` / ``[...]`` region found
    inside surrounding prose. Each candidate is run through tolerant fixes
    before a strict decode attempt.
    """
    if not text:
        raise ValueError("Empty model output; expected structured content")

    candidates: list[str] = []
    tagged = _TAG_RE.search(text)
    if tagged:
        candidates.append(tagged.group(1))
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    stripped = text.strip()
    candidates.append(stripped)
    candidates.append(_find_balanced_region(stripped))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for fix in (_tolerant_fixes, lambda s: s):
            try:
                return json.loads(fix(candidate))
            except json.JSONDecodeError:
                continue
    raise ValueError("Model output contained no parseable structured content")


def _tolerant_fixes(text: str) -> str:
    """Apply light, well-scoped fixes that make model output JSON-compatible."""
    fixed = text
    # Strip a BOM / surrounding whitespace.
    fixed = fixed.strip().lstrip("\ufeff")
    # Replace Python-style literals, but only when they are bare words.
    for py, js in _PY_LITERALS.items():
        fixed = re.sub(rf"\b{py}\b", js, fixed)
    # Single-quoted object keys.
    fixed = _SINGLE_QUOTE_KEY_RE.sub(r'"\1":', fixed)
    # Single-quoted string values (avoid apostrophes inside sentences by only
    # touching the shortest single-quoted run adjacent to ':').
    fixed = _SINGLE_QUOTE_VALUE_RE.sub(r':"\1"', fixed)
    # Trailing commas before } or ].
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    # Unquoted JSON keys like {title: "x"}.
    fixed = re.sub(r"(?<![:\w\"])([A-Za-z_][A-Za-z0-9_]*)\s*:", r'"\1":', fixed)
    return fixed


def _find_balanced_region(text: str) -> str:
    """Return the first balanced ``{...}`` or ``[...]`` region in ``text``."""
    openers = {"{": "}", "[": "]"}
    for idx, ch in enumerate(text):
        if ch not in openers:
            continue
        closer = openers[ch]
        depth = 0
        in_str = False
        escaped = False
        for i in range(idx, len(text)):
            c = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == ch:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[idx : i + 1]
        # If a region is unterminated, fall through and try the next opener.
    return ""
