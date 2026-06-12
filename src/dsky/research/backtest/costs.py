"""Cost model: commissions, slippage, and fees applied per fill.

Every fill in the backtest is charged a total cost of::

    cost = commission_per_share * |shares|
         + (slippage_bps / 10_000) * |notional|

The cost is symmetric: applied on both entry and exit. The default
values are conservative for US equities: $0.005 per share + 5 bps of
notional. Override at construction time for other markets.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-trade cost model. Symmetric; applied on entry and exit.

    Parameters
    ----------
    commission_per_share:
        Flat per-share commission, in the same currency as ``price``.
        Default: ``0.005`` (half a cent per share).
    slippage_bps:
        Slippage + spread model as basis points of notional.
        ``5.0`` = 0.05% of the trade's notional value.
        Default: ``5.0`` (a conservative retail figure for liquid
        US equities).

    Examples
    --------
    >>> c = CostModel()
    >>> # Buy 100 shares of SPY at $450.00:
    >>> round(c.apply(450.0, 100.0), 4)
    2.725

    """

    commission_per_share: float = 0.005
    slippage_bps: float = 5.0

    def apply(self, price: float, shares: float) -> float:
        """Return the total cost for a fill of ``shares`` shares at ``price``.

        Always non-negative. The cost is the same whether the fill
        is an entry or an exit; sign of ``shares`` is irrelevant.
        Returns ``0.0`` if either input is zero.
        """
        if shares == 0.0 or price == 0.0:
            return 0.0
        notional = abs(price * shares)
        commission = self.commission_per_share * abs(shares)
        slippage = notional * (self.slippage_bps / 10_000.0)
        return commission + slippage


__all__ = ["CostModel"]
