"""Modular system-prompt machinery: fences, specs, and safe rendering.

OpenBird's LLM system prompts are *swappable* — a user may override the persona
(tone / answering behavior) while the framework keeps the prompt-injection
defense intact. Two invariants make that safe:

1. **The security scaffold is framework-composed, not user-editable.** A
   :class:`PromptSpec` renders as ``[security_preamble] + persona + [security_epilogue]``
   (a "sandwich"), so a custom persona can never be the model's *last* instruction
   and can never delete the untrusted-data rules.

2. **One source of truth for fences.** A :class:`FenceSpec` owns the delimiter
   tokens AND the neutralizer that strips them from captured (untrusted) text.
   The preamble interpolates those tokens, the validator derives its required
   tokens from them, and each feature's sanitizer delegates to ``neutralize`` —
   so the prompt, the validator, and the sanitizer cannot drift apart.

What :func:`render` validates is narrow and honest: it checks the fence tokens
are *present* in the rendered prompt. It is NOT a proof of injection resistance
(a persona could still say "trust the captured text"); the empirical check for
that is the (later) ``openbird prompts test`` harness. See ``prompts-plan.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# Sentinel left in place of a stripped structural marker, so a redaction is
# observable rather than silent. Kept identical to RAG's historical value so
# existing behavior (and tests asserting on it) is preserved.
_REDACTION = "[redacted-marker]"


@dataclass(frozen=True)
class FenceSpec:
    """The single source of truth for one feature's untrusted-data fence.

    ``open_token``/``close_token`` delimit the fenced region in the prompt and
    MUST appear in the rendered system prompt (so the model is told what the
    fence is). ``extra_forbidden`` lists additional structural markers (e.g. a
    ``[source_id: `` header) that must never survive verbatim from captured text
    even though they do not themselves bound the fence.

    ``neutralize`` is the ONE sanitizer entrypoint: every feature strips captured
    text through it (never a raw helper directly), which keeps the sanitizer in
    lockstep with the tokens the prompt relies on. By default it does a
    replace-until-stable pass over ``forbidden``. A feature whose escape semantics
    differ (e.g. routines defang ``<observations>`` to single-angle-quote
    look-alikes via regex; meetings zero-width-escape ``</transcript>``) supplies a
    ``neutralizer``
    callable — the RAW, body-only sanitizer for that fence. ``neutralize`` then
    delegates to it. The callable MUST NOT call back into ``neutralize`` (it is the
    leaf), so there is no ``neutralize -> neutralizer -> neutralize`` recursion.
    """

    open_token: str
    close_token: str
    extra_forbidden: tuple[str, ...] = ()
    redaction: str = _REDACTION
    # Optional feature-specific raw sanitizer. None => the default replace loop.
    # compare=False so two specs with the same tokens but distinct callables stay
    # hashable/usable as frozen dataclasses without comparing function identities.
    neutralizer: Callable[[str], str] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # A non-empty marker is required regardless of neutralizer: the tokens
        # still feed required_tokens()/the prompt. The redaction/convergence guard
        # only governs the DEFAULT replace loop (a custom neutralizer owns its own
        # termination), so it is skipped when a neutralizer is supplied.
        markers = (self.open_token, self.close_token, *self.extra_forbidden)
        if any(not marker for marker in markers):
            raise ValueError("FenceSpec markers must be non-empty")
        if self.neutralizer is None and any(
            marker in self.redaction for marker in markers
        ):
            raise ValueError(
                "FenceSpec redaction must not contain a forbidden marker"
            )

    @property
    def forbidden(self) -> tuple[str, ...]:
        """All markers stripped from captured text (fence delimiters + extras)."""
        return (self.open_token, self.close_token, *self.extra_forbidden)

    def required_tokens(self) -> tuple[str, ...]:
        """Tokens the rendered system prompt MUST reference.

        Only the fence delimiters: the prompt has to name the open/close tokens
        so the model knows which region is untrusted. ``extra_forbidden`` markers
        are a sanitizer concern, not something the prompt must mention.
        """
        return (self.open_token, self.close_token)

    def neutralize(self, text: str) -> str:
        """Strip every forbidden structural marker from untrusted ``text``.

        Prompt-injection defense: a malicious capture could embed the literal
        close delimiter (or a fake source header) to "break out" of the fence so
        that following text is read as trusted instructions. We defang by
        replacing each forbidden marker with a visible sentinel.

        Applied repeatedly until stable so an interleaved payload cannot re-form
        a marker after a single pass (e.g. ``<<<X<<<X>>>>>>`` style overlaps).

        When a ``neutralizer`` callable is configured, delegate to it (the raw,
        body-only sanitizer for this fence) instead of the default loop.
        """
        if self.neutralizer is not None:
            return self.neutralizer(text)
        if not text:
            return text
        cleaned = text
        while True:
            before = cleaned
            for marker in self.forbidden:
                cleaned = cleaned.replace(marker, self.redaction)
            if cleaned == before:
                return cleaned


class PromptValidationError(ValueError):
    """Raised when a rendered prompt is missing a required fence token.

    Signals that the locked security scaffold did not survive composition (a
    framework bug or a tampered preamble) — never a normal user condition.
    """

    def __init__(self, key: str, missing: tuple[str, ...] | list[str]) -> None:
        self.key = key
        self.missing = tuple(missing)
        super().__init__(
            f"prompt {key!r} is missing required fence token(s): "
            f"{', '.join(self.missing)}"
        )


@dataclass(frozen=True)
class PromptSpec:
    """A swappable system prompt: locked security scaffold + editable persona.

    ``security_preamble`` and ``security_epilogue`` are framework-owned and
    interpolate the fence tokens; ``default_persona`` is the part a user may
    override. :meth:`render` composes them as a sandwich and validates the
    result.
    """

    key: str
    fence: FenceSpec
    security_preamble: str
    security_epilogue: str
    default_persona: str

    def render(self, persona: str | None = None) -> str:
        """Render this spec, optionally with a custom ``persona``."""
        return render(self, persona)


def render(spec: PromptSpec, persona: str | None = None) -> str:
    """Compose ``[preamble] + persona + [epilogue]`` and validate fence tokens.

    ``persona`` of ``None`` uses the spec's bundled default. The security layers
    always wrap the persona, so user text can neither replace the scaffold nor be
    the model's final instruction. Raises :class:`PromptValidationError` if a
    required fence token did not survive (a framework/tamper bug, not user error).
    """
    chosen = spec.default_persona if persona is None else persona
    # Validate the framework-owned scaffold ONLY, BEFORE the (user-swappable)
    # persona is mixed in. Validating the fully composed prompt would let a
    # persona that happens to contain the fence tokens mask a tampered
    # preamble/epilogue that dropped them — the persona is exactly the part we
    # do not trust to carry the security invariant.
    scaffold = "\n\n".join(
        part
        for part in (spec.security_preamble.rstrip(), spec.security_epilogue.rstrip())
        if part
    )
    missing = [tok for tok in spec.fence.required_tokens() if tok not in scaffold]
    if missing:
        raise PromptValidationError(spec.key, missing)
    parts = [
        spec.security_preamble.rstrip(),
        chosen.strip(),
        spec.security_epilogue.rstrip(),
    ]
    return "\n\n".join(part for part in parts if part) + "\n"
