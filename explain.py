"""
ScamShield - plain-language alert summaries.

An analyst has seconds to decide.  A list of SHAP numbers is useless to them;
one sentence in plain English is not.  This module turns the top contributing
features (from XGBoost's built-in TreeSHAP, pred_contribs=True - no extra
package needed) into something a human can act on, e.g.

    "Account drained to zero; recipient received money from 14 different
     senders in the last 12 hours (possible mule); money forwarded on within
     2 hours of arriving."

Everything here is template based and deterministic: no network call, no extra
model, nothing that can add latency to a real-time decision.  The optional
generative hook at the bottom is OFF by default.
"""
from __future__ import annotations

from typing import Any, Sequence

import config
from features import FEATURE_NAMES

# ----------------------------------------------------------------------
# One template per feature.  Each entry is (phrase_when_raising_risk,
# phrase_when_lowering_risk).  Either may be None if that direction is not
# worth telling a human about.  "{v}" is replaced with the feature's value.
# ----------------------------------------------------------------------
_TEMPLATES: dict[str, tuple[str | None, str | None]] = {
    "amount": ("unusually large amount of {money}", "small amount ({money})"),
    "log_amount": ("unusually large amount of {money}", "small amount ({money})"),
    "amount_to_balance_ratio": (
        "transfer is {pct} of the whole account balance",
        "transfer is only a small slice of the balance",
    ),
    "hour_of_day": ("sent at {hour}:00, an unusual hour for this account", None),
    "type_CASH_IN": (None, "ordinary cash-in"),
    "type_CASH_OUT": ("cash-out, the usual last hop before money disappears", None),
    "type_DEBIT": ("card debit", None),
    "type_PAYMENT": (None, "ordinary merchant payment"),
    "type_TRANSFER": ("account-to-account transfer", None),
    "balance_error_orig": (
        "sender's ledger does not add up (off by {money})",
        None,
    ),
    "balance_error_dest": (
        "recipient's ledger does not add up (off by {money})",
        None,
    ),
    "orig_drained_to_zero": ("account drained to zero", None),
    "dest_was_empty": ("recipient account was empty before this payment", None),
    "dest_fanin_senders": (
        "recipient received money from {int_v} different senders in the last "
        f"{config.FANIN_WINDOW_STEPS} hours (possible mule)",
        "recipient is a familiar, low-traffic account",
    ),
    "dest_new_sender": ("first time this sender has ever paid this recipient", None),
    "dest_rapid_passthrough": (
        "recipient has a history of forwarding money straight back out", None,
    ),
    "orig_rapid_passthrough": (
        "money is being forwarded on within "
        f"{config.PASSTHROUGH_WINDOW_STEPS} hours of arriving (pass-through)",
        None,
    ),
    "orig_prior_txn_count": (
        "sender has almost no transaction history",
        "sender has a long, normal history",
    ),
    "dest_prior_inflow_count": (
        "recipient is brand new to the network", "recipient is well established",
    ),
    # --- mule ring / network ------------------------------------------------
    "dest_ring_size": (
        "recipient belongs to a cluster of {int_v} accounts that pass money "
        "between themselves (mule network)",
        None,
    ),
    "orig_ring_size": (
        "sender belongs to a {int_v}-account pass-through cluster", None,
    ),
    "dest_ring_volume": (
        "the recipient's cluster of accounts has already moved {money} "
        "between its members",
        None,
    ),
    # --- velocity / structuring --------------------------------------------
    "orig_txns_in_window": (
        "sender made {int_v} payments in the last "
        f"{config.VELOCITY_WINDOW_STEPS} hours (unusual burst)",
        "sender's payment rate is normal",
    ),
    "orig_amount_in_window": (
        "sender has moved {money} in the last "
        f"{config.VELOCITY_WINDOW_STEPS} hours",
        None,
    ),
    "amount_just_below_round": (
        "amount sits just under a reporting threshold - the classic sign of a "
        "large transfer being split up (structuring)",
        None,
    ),
    # --- victim side -------------------------------------------------------
    "orig_amount_vs_mean": (
        "payment is {mult} this customer's usual amount - out of character",
        "amount is in line with this customer's normal behaviour",
    ),
    "orig_amount_vs_max": (
        "largest payment this customer has ever made", None,
    ),
    "orig_first_time_night": (
        "first payment this customer has ever made during the night", None,
    ),
}

# Features whose *value* is money and should be formatted as currency.
_MONEY_FEATURES = {"amount", "log_amount", "balance_error_orig", "balance_error_dest"}

# Signals that only mean something above a floor.  A "cluster of 1 account" or
# a payment 1.1x the usual size is noise, and putting noise in front of an
# analyst is how they learn to ignore the queue.
_MIN_VALUE_TO_MENTION: dict[str, float] = {
    "dest_ring_size": 3.0,
    "orig_ring_size": 3.0,
    "dest_ring_volume": 1.0,
    "orig_txns_in_window": 4.0,
    "orig_amount_vs_mean": 3.0,
    "orig_amount_vs_max": 1.0,
}


def format_money(value: float) -> str:
    """Compact currency rendering, e.g. 1234567 -> '1.23M'."""
    v = abs(float(value))
    sign = "-" if value < 0 else ""
    if v >= 1_000_000:
        return f"{sign}{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{sign}{v / 1_000:.1f}K"
    return f"{sign}{v:,.0f}"


def _render(name: str, value: float, contribution: float) -> str | None:
    """Render one feature's template, or None if it is not worth saying."""
    raising, lowering = _TEMPLATES.get(name, (None, None))
    template = raising if contribution >= 0 else lowering
    if not template:
        return None

    # One-hot and flag features only make sense when they are actually set.
    if name.startswith("type_") and value <= 0:
        return None
    if name in ("orig_drained_to_zero", "dest_was_empty", "dest_new_sender",
                "dest_rapid_passthrough", "orig_rapid_passthrough",
                "amount_just_below_round", "orig_first_time_night") and value <= 0:
        return None

    # Do not report a signal that is technically present but trivially small.
    floor = _MIN_VALUE_TO_MENTION.get(name)
    if floor is not None and contribution >= 0 and value < floor:
        return None

    money = format_money(value if name not in ("log_amount",) else (2.718281828 ** value) - 1)
    return template.format(
        v=f"{value:,.2f}",
        int_v=int(round(value)),
        money=money,
        pct=f"{min(value, 9.99) * 100:.0f}%",
        mult=f"{value:,.0f}x" if value >= 2 else f"{value:.1f}x",
        hour=int(value) % 24,
    )


def top_contributions(
    contribs: Sequence[float],
    feature_values: Sequence[float],
    k: int = config.TOP_REASONS,
) -> list[dict[str, Any]]:
    """
    Pick the k features that pushed the score UP the most.

    contribs: one TreeSHAP value per feature (the trailing bias term that
              XGBoost appends must already be stripped by the caller).
    """
    pairs = [
        {
            "feature": FEATURE_NAMES[i],
            "value": float(feature_values[i]),
            "contribution": float(contribs[i]),
        }
        for i in range(min(len(FEATURE_NAMES), len(contribs)))
    ]
    pairs.sort(key=lambda d: d["contribution"], reverse=True)
    return pairs[:k]


def summarise(reasons: Sequence[dict[str, Any]], score: float | None = None) -> str:
    """Turn the top contributions into one human sentence."""
    if config.USE_GENERATIVE_SUMMARY:
        generated = _generative_summary(reasons, score)
        if generated:
            return generated

    phrases: list[str] = []
    for r in reasons:
        text = _render(r["feature"], r["value"], r["contribution"])
        if text and text not in phrases:
            phrases.append(text)

    if not phrases:
        return "No single dominant risk factor - flagged on the combined pattern."

    sentence = "; ".join(phrases)
    return sentence[0].upper() + sentence[1:] + "."


def explain(
    contribs: Sequence[float],
    feature_values: Sequence[float],
    score: float | None = None,
    k: int = config.TOP_REASONS,
) -> tuple[list[dict[str, Any]], str]:
    """Convenience: top-k reasons plus the plain-language sentence."""
    reasons = top_contributions(contribs, feature_values, k)
    return reasons, summarise(reasons, score)


# ----------------------------------------------------------------------
# OPTIONAL generative hook - DISABLED BY DEFAULT
# ----------------------------------------------------------------------
# Set config.USE_GENERATIVE_SUMMARY = True and fill this in to have an LLM
# write the analyst-facing sentence instead of the templates above.
#
# Deliberately left unimplemented for the hackathon demo because:
#   * it adds network latency to a path that must stay under a millisecond,
#   * it is non-deterministic, which is a problem for an audit trail,
#   * the container on LinuxONE has no outbound model endpoint.
#
# If you do enable it, keep the template output as the fallback (as below) so
# the dashboard never shows an empty reason.
def _generative_summary(reasons: Sequence[dict[str, Any]], score: float | None) -> str | None:
    """Return a generated sentence, or None to fall back to the templates."""
    # Example shape of an implementation:
    #
    #   prompt = ("Explain to a bank fraud analyst, in one sentence, why this "
    #             "payment was flagged. Factors: "
    #             + ", ".join(f"{r['feature']}={r['value']:.2f}" for r in reasons))
    #   return my_llm_client.complete(prompt)
    #
    return None
