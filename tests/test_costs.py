"""Unit tests for the cost model.

The cost model is the only place where commissions and slippage live.
If the formula changes, these tests must change in lockstep -- the
backtest engine's net-PnL arithmetic depends on this contract.
"""
import pytest

from dsky.research.backtest.costs import CostModel


class TestCostModelApply:
    """The total cost for a fill."""

    def test_zero_shares_returns_zero(self) -> None:
        assert CostModel().apply(price=100.0, shares=0.0) == 0.0

    def test_zero_price_returns_zero(self) -> None:
        assert CostModel().apply(price=0.0, shares=100.0) == 0.0

    def test_commission_only_when_slippage_is_zero(self) -> None:
        c = CostModel(commission_per_share=0.01, slippage_bps=0.0)
        # 100 shares * $0.01 = $1.00
        assert c.apply(price=50.0, shares=100.0) == pytest.approx(1.0)

    def test_slippage_only_when_commission_is_zero(self) -> None:
        c = CostModel(commission_per_share=0.0, slippage_bps=10.0)
        # 10 bps of $10,000 notional = $10.00
        assert c.apply(price=100.0, shares=100.0) == pytest.approx(10.0)

    def test_combined_commission_and_slippage(self) -> None:
        c = CostModel(commission_per_share=0.005, slippage_bps=5.0)
        # 100 shares * $0.005 = $0.50
        # 5 bps of $45,000 notional = $22.50
        # Total: $23.00
        assert c.apply(price=450.0, shares=100.0) == pytest.approx(23.0)

    def test_short_side_is_symmetric(self) -> None:
        """A short fill of -100 shares costs the same as a long fill of +100."""
        c = CostModel()
        long_cost = c.apply(price=50.0, shares=100.0)
        short_cost = c.apply(price=50.0, shares=-100.0)
        assert long_cost == short_cost

    def test_cost_is_always_non_negative(self) -> None:
        c = CostModel()
        for shares in (100.0, -100.0, 0.001, -0.001):
            assert c.apply(price=100.0, shares=shares) >= 0.0

    def test_default_values_are_conservative(self) -> None:
        """Defaults are US-equities-like: half-cent/share + 5 bps."""
        c = CostModel()
        assert c.commission_per_share == pytest.approx(0.005)
        assert c.slippage_bps == pytest.approx(5.0)

    def test_cost_scales_linearly_with_shares(self) -> None:
        """Doubling the share count doubles the total cost."""
        c = CostModel()
        one_lot = c.apply(price=100.0, shares=100.0)
        two_lot = c.apply(price=100.0, shares=200.0)
        assert two_lot == pytest.approx(2 * one_lot)

    def test_slippage_scales_linearly_with_price(self) -> None:
        """Doubling the price doubles the slippage component (notional doubles)."""
        c = CostModel(commission_per_share=0.0, slippage_bps=5.0)
        cheap = c.apply(price=50.0, shares=100.0)
        pricey = c.apply(price=100.0, shares=100.0)
        assert pricey == pytest.approx(2 * cheap)


class TestCostModelFrozen:
    """The cost model is immutable -- it's a parameter, not state."""

    def test_cost_model_is_frozen(self) -> None:
        c = CostModel()
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError on stdlib
            c.commission_per_share = 0.0  # type: ignore[misc]
