"""Token-level plans independent of vLLM scheduling and GPU execution."""
from dataclasses import dataclass
from .kvcache import KVGraft, KVPrefix, longest_common_prefix
from .repair import repair_token_counts


@dataclass(frozen=True)
class ReuseSpan:
    start: int
    end: int
    graft: KVGraft
    source_offset: int


@dataclass
class PrefillPlan:
    prefix: KVPrefix | None
    prefix_length: int
    spans: list[ReuseSpan]
    mismatch: bool = False

    @property
    def reused_tokens(self):
        return self.prefix_length + sum(s.end - s.start for s in self.spans)

    @property
    def exact_prefix_length(self):
        return self.spans[0].start if self.spans else None


def plan_prefill(tokens, prefixes=(), grafts=(), *, begin=0.0, end=0.0, exact=False):
    """Validate all grafts first; fall back to exact LCP on mismatch.

    Never skip the final query. Approximate KV cannot become an exact hit.
    """
    if not tokens:
        raise ValueError("prompt must not be empty")
    grafts = sorted(grafts or (), key=lambda g: g.position)
    previous_end, mismatch = 0, False
    for g in grafts:
        stop = g.position + len(g.tokens)
        if not (0 < g.position < stop <= len(tokens)
                and previous_end <= g.position
                and tokens[g.position:stop] == g.tokens
                and g.source_position >= 0
                and g.cache.get_seq_length() == len(g.tokens)):
            mismatch = True
            break
        previous_end = stop
    if mismatch or exact:
        grafts = []
    spans = []
    for g in grafts:
        left, right = repair_token_counts(len(g.tokens), begin, end)
        start = g.position + left
        stop = min(g.position + len(g.tokens) - right, len(tokens) - 1)
        if start < stop:
            spans.append(ReuseSpan(start, stop, g, left))
    cap = min(len(tokens) - 1, spans[0].start if spans else len(tokens))
    best, length = None, 0
    for candidate in prefixes or ():
        matched = min(longest_common_prefix(tokens, candidate.tokens),
                      candidate.exact_length, candidate.cache.get_seq_length(), cap)
        if matched > length:
            best, length = candidate, matched
    return PrefillPlan(best, length, spans, mismatch)
