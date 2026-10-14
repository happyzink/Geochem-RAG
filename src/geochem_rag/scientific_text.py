"""Display-side rendering that preserves scientific literals.

The user reported numbers, decimal points and scientific symbols being lost or
turned into bars. Layer-by-layer tracing (task A) separated two causes:

1. Extraction/ingest layer: some PDFs (broken font encodings) yield control
   characters or full-width glyphs. That is a data problem, not a display one;
   those pages are quality-flagged and handled in the ingest pipeline.
2. **Render layer**: Streamlit's ``st.markdown`` uses GFM, whose strikethrough
   plugin (``singleTilde``) pairs ANY two ``~`` characters. Scientific text such
   as ``251~16 Ma`` and ``0.9~1.9`` therefore renders a strike-through line that
   visually looks like a bar through the digits. The characters were never
   replaced in storage — they were hidden by markdown parsing.

This module fixes layer 2 only. It escapes bare ``~`` for a **display copy** so
GFM never treats scientific tildes as strikethrough, while:

- code spans (`` `...` ``) and math (``$...$``) are copied verbatim, so `` `Sr/Y` ``
  and ``$10^{-3}$`` keep their exact content;
- an already-escaped ``\\~`` is left untouched, so replaying / re-rendering a
  message never accumulates backslashes;
- a genuine ``~~pair~~`` (an even run of two or more tildes) is kept, so
  intentional strikethrough still works and no character is silently changed.

It never rewrites ``-``, ``−``, ``–``, punctuation or digits; it does not touch
the stored evidence text or any hash. Call it only on the text that is about to
be shown with ``st.markdown``.
"""

from __future__ import annotations

_TILDE = "~"
_CODE = "`"
_MATH = "$"


def _escape_plain(segment: str) -> str:
    """Escape bare tildes in a markdown-plain segment (idempotent)."""
    if _TILDE not in segment:
        return segment
    out: list[str] = []
    i = 0
    n = len(segment)
    while i < n:
        if segment[i] != _TILDE:
            out.append(segment[i])
            i += 1
            continue
        j = i
        while j < n and segment[j] == _TILDE:
            j += 1
        run_len = j - i
        already_escaped = i > 0 and segment[i - 1] == "\\"
        if already_escaped or (run_len >= 2 and run_len % 2 == 0):
            # Keep the run as-is: either it was already escaped by us before,
            # or it is an even ``~~`` pair the author intended as strikethrough.
            out.append(segment[i:j])
        else:
            # Escape every tilde in an odd/singleton run so GFM cannot pair it.
            out.append("\\~" * run_len)
        i = j
    return "".join(out)


def escape_tilde_for_markdown(text: str) -> str:
    """Return a display copy with scientific tildes protected from GFM.

    Safe to call repeatedly on the same string: the result is stable.
    """
    if not isinstance(text, str) or _TILDE not in text:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == _CODE:
            j = i
            while j < n and text[j] == _CODE:
                j += 1
            ticks = j - i
            closing = text.find(_CODE * ticks, j)
            if closing != -1:
                end = closing + ticks
                out.append(text[i:end])  # verbatim: code spans are not parsed
                i = end
                continue
            out.append(ch)
            i += 1
            continue
        if ch == _MATH:
            if text.startswith("$$", i):
                closing = text.find("$$", i + 2)
                if closing != -1:
                    out.append(text[i : closing + 2])
                    i = closing + 2
                    continue
            else:
                closing = text.find(_MATH, i + 1)
                newline = text.find("\n", i + 1)
                if closing != -1 and (newline == -1 or closing < newline):
                    out.append(text[i : closing + 1])  # verbatim math
                    i = closing + 1
                    continue
            out.append(ch)
            i += 1
            continue
        j = i
        while j < n and text[j] not in (_CODE, _MATH):
            j += 1
        out.append(_escape_plain(text[i:j]))
        i = j
    return "".join(out)


def safe_markdown(text: str) -> str:
    """Alias used by the app for any model/corpus text shown via st.markdown."""
    return escape_tilde_for_markdown(text)


def contains_unescaped_tilde(text: str) -> bool:
    """True if ``text`` still exposes a pair-able bare or odd ``~`` run.

    Operates on a text that has not been through
    :func:`escape_tilde_for_markdown`, so it reports whether the markdown
    layer would treat a scientific tilde as strikethrough. Even ``~~`` runs
    (intentional strikethrough) and code/math spans are ignored.
    """
    stripped = text
    # After escaping, every remaining ``~`` must belong to an even ``~~`` run
    # (intentional strikethrough) or sit inside a protected code/math span.
    i = 0
    n = len(stripped)
    while i < n:
        ch = stripped[i]
        if ch == _CODE:
            j = i
            while j < n and stripped[j] == _CODE:
                j += 1
            ticks = j - i
            closing = stripped.find(_CODE * ticks, j)
            i = (closing + ticks) if closing != -1 else i + 1
            continue
        if ch == _MATH:
            if stripped.startswith("$$", i):
                closing = stripped.find("$$", i + 2)
                i = (closing + 2) if closing != -1 else i + 1
            else:
                closing = stripped.find(_MATH, i + 1)
                newline = stripped.find("\n", i + 1)
                i = (closing + 1) if (closing != -1 and (newline == -1 or closing < newline)) else i + 1
            continue
        if ch == _TILDE:
            if i > 0 and stripped[i - 1] == "\\":
                i += 1
                continue
            j = i
            while j < n and stripped[j] == _TILDE:
                j += 1
            if (j - i) % 2 != 0:
                return True
            i = j
            continue
        i += 1
    return False
