"""Outlier sensitivity: recompute metric after dropping top-2 trades."""

from dataclasses import dataclass

from dsky.research.backtest.engine import Fill
from dsky.research.robustness._math import round_trip_contributions

# Edge is considered outlier-dependent when removing the top-2
# contributing round trips leaves less than this fraction of the
# original metric (or the residual is negligible while the headline
# metric was material).
_RETAINED_FRACTION = 0.25
_MATERIAL_EDGE = 0.005
_MIN_TRADES_FOR_DROP = 2


@dataclass(frozen=True)
class OutlierDropResult:
    """Metric after dropping the two largest-contribution round trips."""

    metric_after_drop: float
    dropped_contribution: float
    edge_lost: bool


def analyze_outlier_drop(
    *,
    key_metric: float,
    trades: list[Fill],
    position_size: float,
) -> OutlierDropResult:
    """Recompute the key metric without the top-2 contributing trades.

    Flags ``edge_lost`` when the residual metric is a small fraction of
    the headline metric -- the edge depended on one or two outlier fills.
    """
    contribs = round_trip_contributions(trades, position_size=position_size)
    if len(contribs) < _MIN_TRADES_FOR_DROP:
        return OutlierDropResult(
            metric_after_drop=key_metric,
            dropped_contribution=0.0,
            edge_lost=False,
        )

    sorted_pnls = sorted((pnl for _, pnl in contribs), reverse=True)
    top_two = sorted_pnls[0] + sorted_pnls[1]
    metric_after = key_metric - top_two
    edge_lost = False
    if abs(key_metric) >= _MATERIAL_EDGE:
        retained_ratio = abs(metric_after) / abs(key_metric) if key_metric != 0 else 0.0
        edge_lost = retained_ratio < _RETAINED_FRACTION

    return OutlierDropResult(
        metric_after_drop=metric_after,
        dropped_contribution=top_two,
        edge_lost=edge_lost,
    )


__all__ = ["OutlierDropResult", "analyze_outlier_drop"]
