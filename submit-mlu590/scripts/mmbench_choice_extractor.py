"""Conservative, gold-blind extraction of one MMBench option from a reply.

The extractor only receives the response, its finish reason, and available
option letters. It deliberately leaves unsupported or conflicting prose as Z.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


RULE_VERSION = "mmbench_choice_v3.0.0"

# Remove presentation markup for recognition; the original text remains in the
# per-question audit. NFKC also covers full-width Latin letters and colons.
_MARKDOWN = str.maketrans("", "", "*_`")
_EXPLICIT = re.compile(
    r"""(?ix)
    (?P<cue>
        \b(?:the\s+)?(?:final|correct|best)\s+(?:answer|option|choice)\b
      | \b(?:the|my)\s+(?:answer|option|choice)\b
      | \banswer\b(?=\s*(?:is\b|[:=]))
      | (?:最终|正确)?答案
      | 我(?:选择|选)
    )
    \s*(?:(?:is|would\s+be|为|是|选)\s*)?
    (?::|=)?\s*(?:(?:option|choice|选项)\s*)?
    [\(\[\{]?\s*(?P<option>[A-D])(?=$|[^\w]|项)
    """
)
_REVERSE_EXPLICIT = re.compile(
    r"""(?ix)
    \b(?:(?:option|choice)\s+)?(?P<option>[A-D])\s+
    (?:is|was)\s+(?!not\b|never\b)
    (?:(?:the|a)\s+)?(?:(?:correct|final|best)\s+answer|answer|
      (?:(?:also|equally)\s+)?(?:correct|right|valid|true))\b
    """
)
_REVERSE_EXPLICIT_CN = re.compile(r"(?i)\b(?P<option>[A-D])\s*(?:是|为)\s*(?:正确|最终)?答案")
_DIRECT_ALTERNATIVE = re.compile(
    r"(?ix)^\s*(?:/|,|、|\bor\b|\band\b|或|和|与)\s*(?:(?:option|choice)\s*)?([A-D])\b"
)
_MULTIPLE_CORRECT = (
    re.compile(
        r"(?ix)\b(?:both|either)\s+(?:options?\s+)?([A-D])\s+(?:and|or)\s+"
        r"(?:options?\s+)?([A-D])\s+(?:(?:are|is|could\s+be)\s+)?"
        r"(?:(?:both|equally)\s+)?(?:correct|right|valid|acceptable|true)\b"
    ),
    re.compile(
        r"(?ix)\b(?:options?\s+)?([A-D])\s+(?:and|or)\s+(?:options?\s+)?([A-D])\s+"
        r"(?:are|is|could\s+be)\s+(?:(?:both|equally)\s+)?"
        r"(?:correct|right|valid|acceptable|true)\b"
    ),
)
_MULTIPLE_CORRECT_CN = re.compile(
    r"(?i)\b([A-D])\s*(?:和|与|及|、|或)\s*([A-D])\s*(?:都|均)?\s*(?:是)?\s*(?:正确|对的|对|可选)"
)
_WHOLE_OPTION = re.compile(r"(?ix)^\s*(?:(?:option|choice)\s+)?[\(\[]?([A-D])[\)\]]?[.:]?\s*$")
_STANDALONE_LINE = re.compile(
    r"(?im)^\s*(?:[-+]\s*)?(?:(?:option|choice)\s+)?[\(\[]?([A-D])[\)\]]?[.:]?\s*$"
)
_ANY_OPTION_LABEL = re.compile(r"(?i)(?<!\w)([A-D])[.):]\s+\S")
_LEADING_LABEL = re.compile(r"(?i)^\s*(?:#{1,6}\s*)?(?:[-+]\s*)?([A-D])[.):]\s+\S")
_TERMINAL_LABEL = re.compile(r"(?i)^\s*\*\*\s*([A-D])[.):]\s+[^*\n]+\*\*\s*$")
_TERMINAL_CUE = re.compile(r"(?i)\b(?:therefore|thus|hence|in\s+conclusion|所以|因此|综上)\b")
_NEGATIVE_CUE = re.compile(r"(?i)\b(?:wrong|incorrect|false|rejected|not\s+the)\b")
_BOLD_INLINE_OPTION = re.compile(r"(?i)\*\*\s*([A-D])\.\s*([^*\n]+?)\s*\*\*")
_COMPARATIVE_IMAGE_QUESTION = re.compile(
    r"(?i)\bwhich\s+image\s+is\s+more\s+([a-z][a-z-]*)\s*\?"
)


@dataclass(frozen=True)
class Extraction:
    option: str
    rule: str
    evidence: str | None
    explicit_claims: tuple[dict[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _result(option: str, rule: str, evidence: str | None,
            claims: Iterable[dict[str, object]] = ()) -> Extraction:
    return Extraction(option=option, rule=rule, evidence=evidence,
                      explicit_claims=tuple(claims))


def extract_option(reply: str | None, finish_reason: str | None,
                   available_options: Mapping[str, str] | Iterable[str],
                   question: str | None = None) -> Extraction:
    """Return A-D only for a supported unique conclusion; otherwise return Z.

    No gold answer or option text is inspected. An explicit conclusion overrides
    a leading label, but conflicting conclusions or labels return Z.
    """
    allowed = {str(letter).upper() for letter in available_options}
    if finish_reason != "stop":
        return _result("Z", "incomplete_response", f"finish_reason={finish_reason!r}")
    if not isinstance(reply, str) or not reply.strip():
        return _result("Z", "empty_response", None)

    view = unicodedata.normalize("NFKC", reply).translate(_MARKDOWN)
    claims: list[dict[str, object]] = []
    for match in _EXPLICIT.finditer(view):
        # "wrong answer is A" describes a rejected option, not a conclusion.
        preceding = view[max(0, match.start() - 24):match.start()]
        if re.search(r"(?i)(?:\b(?:wrong|incorrect|false|rejected|not\s+the)\b|错误|不正确|排除|不是)\s*$", preceding):
            continue
        letter = match.group("option").upper()
        claims.append({"option": letter, "cue": match.group("cue"),
                       "evidence": match.group(0).strip(),
                       "span": [match.start(), match.end()]})
        after = view[match.end():]
        alternative = _DIRECT_ALTERNATIVE.match(after)
        if alternative:
            claims.append({"option": alternative.group(1).upper(),
                           "cue": "direct_alternative",
                           "evidence": after[:alternative.end()].strip(),
                           "span": [match.end(), match.end() + alternative.end()]})

    for pattern in (_REVERSE_EXPLICIT, _REVERSE_EXPLICIT_CN):
        for match in pattern.finditer(view):
            claims.append({"option": match.group("option").upper(), "cue": "reverse_conclusion",
                           "evidence": match.group(0).strip(),
                           "span": [match.start(), match.end()]})

    multiple_spans: list[tuple[int, int]] = []
    for pattern in (*_MULTIPLE_CORRECT, _MULTIPLE_CORRECT_CN):
        for match in pattern.finditer(view):
            if any(match.start() < end and start < match.end() for start, end in multiple_spans):
                continue
            multiple_spans.append((match.start(), match.end()))
            for letter in (match.group(1).upper(), match.group(2).upper()):
                claims.append({"option": letter, "cue": "multiple_correct",
                               "evidence": match.group(0).strip(),
                               "span": [match.start(), match.end()]})

    # Overlapping patterns can report the same claim more than once. Keep the
    # evidence compact without changing the set of claimed options.
    claims = list({(c["option"], tuple(c["span"])): c for c in claims}.values())

    if claims:
        letters = {claim["option"] for claim in claims}
        evidence = "; ".join(str(claim["evidence"]) for claim in claims)
        if len(letters) != 1:
            return _result("Z", "conflicting_explicit_conclusions", evidence, claims)
        letter = next(iter(letters))
        if letter not in allowed:
            return _result("Z", "unavailable_explicit_option", evidence, claims)
        return _result(letter, "explicit_conclusion", evidence, claims)

    whole = _WHOLE_OPTION.fullmatch(view)
    if whole:
        letter = whole.group(1).upper()
        return _result(letter if letter in allowed else "Z", "whole_option",
                       whole.group(0).strip())

    # A concluding sentence followed by a final bold option is an answer cue,
    # even if preceding analysis listed several bold options.
    paragraphs = [line.strip() for line in unicodedata.normalize("NFKC", reply).splitlines() if line.strip()]
    if len(paragraphs) >= 2:
        terminal = _TERMINAL_LABEL.fullmatch(paragraphs[-1])
        if (terminal and paragraphs[-2].endswith((":", "："))
                and _TERMINAL_CUE.search(paragraphs[-2])
                and not _NEGATIVE_CUE.search(paragraphs[-2])):
            letter = terminal.group(1).upper()
            if letter in allowed:
                return _result(letter, "concluding_labeled_line", paragraphs[-2] + " " + paragraphs[-1])

    # A bold, labeled *complete option* in the opening answer line is stronger
    # evidence than an incidental letter in subsequent analysis. Verify the
    # label against the option text; neither the gold answer nor image is read.
    if isinstance(available_options, Mapping):
        first_line = next((line.strip() for line in reply.splitlines() if line.strip()), "")
        inline = list(_BOLD_INLINE_OPTION.finditer(first_line))
        if len(inline) > 1:
            return _result("Z", "multiple_opening_inline_options", first_line)
        if len(inline) == 1:
            match = inline[0]
            letter = match.group(1).upper()
            stated_text = match.group(2).strip().rstrip(".。:： ")
            option_text = str(available_options.get(letter, "")).strip().rstrip(".。:： ")
            preceding = first_line[:match.start()]
            if (letter in allowed and stated_text.casefold() == option_text.casefold()
                    and not _NEGATIVE_CUE.search(preceding)
                    and not re.search(r"(?i)\b(?:is|are)\s+not\s*$", preceding)):
                return _result(letter, "opening_bold_exact_option", match.group(0).strip())

        # For a narrow comparative question, an opening sentence that repeats
        # one option text and the queried property is a direct answer. Requiring
        # the whole sentence prevents taking an option mentioned in a contrast.
        comparative = _COMPARATIVE_IMAGE_QUESTION.search(question or "")
        if comparative:
            property_name = comparative.group(1)
            matches = []
            for letter, option_text in available_options.items():
                pattern = re.compile(
                    rf"(?i)^\s*{re.escape(str(option_text).strip())}\s+is\s+more\s+"
                    rf"{re.escape(property_name)}\s*[.!。]?\s*$"
                )
                if pattern.fullmatch(first_line):
                    matches.append(str(letter).upper())
            if len(matches) == 1 and matches[0] in allowed:
                return _result(matches[0], "opening_exact_option_and_question_property", first_line)

    standalone = list(_STANDALONE_LINE.finditer(view))
    leading = _LEADING_LABEL.match(view)
    structured_letters = {match.group(1).upper() for match in standalone}
    if leading:
        structured_letters.add(leading.group(1).upper())
        structured_letters.update(match.group(1).upper() for match in _ANY_OPTION_LABEL.finditer(view))
    if len(structured_letters) > 1:
        return _result("Z", "conflicting_structured_labels", ", ".join(sorted(structured_letters)))
    if standalone:
        letter = standalone[0].group(1).upper()
        if letter in allowed:
            return _result(letter, "standalone_option_line", standalone[0].group(0).strip())
    if leading:
        letter = leading.group(1).upper()
        if letter in allowed:
            return _result(letter, "leading_option_label", view[:leading.end()].strip())

    return _result("Z", "no_unique_option", None)
