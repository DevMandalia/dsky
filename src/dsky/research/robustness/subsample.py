"""Subsample stability: key metric across first/middle/last thirds."""

from dataclasses import dataclass

from dsky.research.backtest.engine import Fill
from dsky.research.robustness._math import metric_from_contributions, round_trip_contributions

# Flag instability when the spread across thirds exceeds this fraction of
# the absolute overall metric, or when thirds straddle zero with material
# magnitude on both sides.
_SPREAD_RATIO = 0.75
_MIN_STRADDLE = 0.001
_MIN_TRADES_FOR_SUBSAMPLE = 3


@dataclass(frozen=True)
class SubsampleResult:
    """Metric reported in each temporal third of the holdout period."""

    first_third_metric: float
    middle_third_metric: float
    last_third_metric: float
    unstable: bool


def analyze_subsample(
    *,
    key_metric: float,
    trades: list[Fill],
    position_size: float,
) -> SubsampleResult:
    """Report the key metric in each third of the trade sequence.

    Splits round trips chronologically into first/middle/last thirds and
    sums each segment's PnL fraction. Flags ``unstable`` when the
    spread across thirds is large relative to the overall metric or
    when profitable and losing thirds coexist with non-trivial size.
    """
    contribs = round_trip_contributions(trades, position_size=position_size)
    if not contribs:
        return SubsampleResult(0.0, 0.0, 0.0, unstable=False)

    n = len(contribs)
    if n < _MIN_TRADES_FOR_SUBSAMPLE:
        # Fewer than three round trips: subsample stability is undefined.
        total = metric_from_contributions(contribs)
        return SubsampleResult(total, 0.0, 0.0, unstable=False)

    third = max(1, n // 3)
    first = contribs[:third]
    middle = contribs[third: 2 * third]
    last = contribs[2 * third:]

    m_first = metric_from_contributions(first)
    m_middle = metric_from_contributions(middle)
    m_last = metric_from_contributions(last)
    thirds = (m_first, m_middle, m_last)
    spread = max(thirds) - min(thirds)
    baseline = max(abs(key_metric), _MIN_STRADDLE)
    sign_straddle = min(thirds) < 0 < max(thirds) and spread >= _MIN_STRADDLE
    unstable = spread > baseline * _SPREAD_RATIO or sign_straddle

    return SubsampleResult(
        first_third_metric=m_first,
        middle_third_metric=m_middle,
        last_third_metric=m_last,
        unstable=unstable,
    )


__all__ = ["SubsampleResult", "analyze_subsample"]
