"""The 14 lifecycle states and the 3 APPROVED sub-types.

State values are the canonical names: CAPTURED, SPECIFIED, PRE_REGISTERED,
BACKTESTED, ROBUSTNESS_CHECKED, ADVERSARIAL_REVIEW, APPROVED,
LIVE_MONITORED, TRIGGERED, THESIS_REVIEW, PAPER_TRADE, ATTRIBUTED,
REJECTED, RETIRED.

The string value of each state is the enum member's name (e.g. State.CAPTURED
has ``.value == "CAPTURED"``). This is the form written into the event log
and stored in the projection.
"""
from enum import StrEnum


class State(StrEnum):
    """The 14 lifecycle states a research idea can occupy.

    The linear forward path is::

        CAPTURED -> SPECIFIED -> PRE_REGISTERED -> BACKTESTED
        -> ROBUSTNESS_CHECKED -> ADVERSARIAL_REVIEW -> APPROVED
        -> LIVE_MONITORED -> TRIGGERED -> THESIS_REVIEW
        -> PAPER_TRADE -> ATTRIBUTED

    Branch exits:
        REJECTED: from any pre-approval research stage
                  (CAPTURED, SPECIFIED, PRE_REGISTERED, BACKTESTED,
                  ROBUSTNESS_CHECKED, ADVERSARIAL_REVIEW).
        RETIRED:  from APPROVED or LIVE_MONITORED.

    Terminal states (no outgoing transitions): ATTRIBUTED, REJECTED, RETIRED.
    """

    CAPTURED = "CAPTURED"
    SPECIFIED = "SPECIFIED"
    PRE_REGISTERED = "PRE_REGISTERED"
    BACKTESTED = "BACKTESTED"
    ROBUSTNESS_CHECKED = "ROBUSTNESS_CHECKED"
    ADVERSARIAL_REVIEW = "ADVERSARIAL_REVIEW"
    APPROVED = "APPROVED"
    LIVE_MONITORED = "LIVE_MONITORED"
    TRIGGERED = "TRIGGERED"
    THESIS_REVIEW = "THESIS_REVIEW"
    PAPER_TRADE = "PAPER_TRADE"
    ATTRIBUTED = "ATTRIBUTED"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


class ApprovalType(StrEnum):
    """The sub-type carried on every APPROVED transition.

    - ``CONTEXT``:       approved as background reading; no live use.
    - ``WATCHLIST``:     approved to be tracked; no live use.
    - ``PAPER_TRADEABLE``: approved to run a paper trade.
    """

    CONTEXT = "context"
    WATCHLIST = "watchlist"
    PAPER_TRADEABLE = "paper_tradeable"
