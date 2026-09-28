"""Budget pacing traffic simulator and evaluation harness.

This module synthesizes a realistic daily traffic curve and then replays a
budget pacing controller against it. It supports the pacing cluster of
ClickRank which maps to the ads pacing and traffic control area.

The traffic curve follows a diurnal pattern. Demand is low overnight, rises
through the morning, peaks in the early evening, and falls again at night. The
run_pacing function drives a pacer slot by slot, records spend, and reports two
headline metrics. Budget utilization is the share of the budget that was spent.
Smoothness is the root mean squared error between the realized cumulative spend
curve and an ideal cumulative spend curve that is proportional to traffic. A
lower smoothness error means the pacer spent in step with demand rather than
dumping budget early.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# One slot per minute across a full day gives 1440 slots by default.
DEFAULT_N_SLOTS: int = 24 * 60


def simulate_traffic(n_slots: int = DEFAULT_N_SLOTS, seed: int = 42) -> np.ndarray:
    """Generate a diurnal eligible impression curve over one day.

    The curve is a smooth sinusoidal base shape that is low overnight and peaks
    in the early evening, multiplied by gentle multiplicative noise so no two
    slots are identical. The result is the count of eligible impressions in each
    slot.

    Parameters
    ----------
    n_slots : int
        Number of time slots in the day. The default is one slot per minute.
    seed : int
        Seed for the noise so the curve is reproducible.

    Returns
    -------
    np.ndarray
        A float64 array of length n_slots with non negative eligible impression
        counts.
    """
    rng = np.random.default_rng(seed)
    # Phase across the day in radians. We place the trough overnight and the
    # peak in the early evening which is the common shape of consumer ad demand.
    t = np.linspace(0.0, 2.0 * np.pi, n_slots, endpoint=False)
    # Peak near hour 18 of 24. Shift the cosine so its maximum lands there.
    peak_fraction = 18.0 / 24.0
    phase = 2.0 * np.pi * peak_fraction
    base_shape = 1.0 + 0.7 * np.cos(t - phase)
    # A secondary midday bump gives a more realistic two hump weekday shape.
    midday_fraction = 12.0 / 24.0
    midday_phase = 2.0 * np.pi * midday_fraction
    base_shape += 0.25 * np.cos(t - midday_phase)
    # Keep the base strictly positive before scaling to a count level.
    base_shape = np.clip(base_shape, 0.05, None)

    mean_eligible_per_slot = 1000.0
    base = base_shape / base_shape.mean() * mean_eligible_per_slot
    # Multiplicative lognormal noise keeps counts positive and adds texture.
    noise = rng.lognormal(mean=0.0, sigma=0.10, size=n_slots)
    traffic = base * noise
    return np.clip(traffic, 0.0, None).astype(np.float64)


def _ideal_spend_curve(traffic: np.ndarray, budget: float) -> np.ndarray:
    """Per slot ideal spend proportional to traffic, summing to the budget.

    The ideal pacer spends in proportion to available demand so that cost per
    mille stays stable and time of day coverage is broad. When there is no
    traffic the ideal curve is flat across slots.
    """
    total = float(np.sum(traffic))
    if total <= 0:
        return np.full(len(traffic), budget / max(len(traffic), 1), dtype=np.float64)
    return traffic / total * budget


def run_pacing(pacer, traffic: np.ndarray, budget: float) -> Dict:
    """Replay a pacer against a traffic curve and score the result.

    The simulator walks the day one slot at a time. For each slot it asks the
    pacer for a throttle probability in the closed interval from 0 to 1 given
    the eligible impressions and the spend so far. The realized spend in that
    slot is the throttle times the eligible impressions, capped by the remaining
    budget so the campaign never overspends. One unit of spend corresponds to
    serving one eligible impression which keeps the units simple.

    Parameters
    ----------
    pacer : object
        Any pacer exposing reset(total_budget, n_slots, traffic) and
        step(slot, eligible, spent_so_far). See src.pacing.controller.
    traffic : np.ndarray
        Per slot eligible impression counts from simulate_traffic.
    budget : float
        The total daily budget in spend units.

    Returns
    -------
    dict
        spend : (n_slots,) float64 realized spend per slot.
        cumulative : (n_slots,) float64 cumulative realized spend.
        throttle : (n_slots,) float64 throttle probability used per slot.
        ideal_spend : (n_slots,) float64 ideal per slot spend.
        ideal_cumulative : (n_slots,) float64 ideal cumulative spend.
        traffic : (n_slots,) float64 the input traffic curve.
        budget : float the total budget.
        budget_utilization : float share of the budget that was spent in [0, 1].
        smoothness_rmse : float root mean squared error of cumulative spend
            against the ideal cumulative curve.
        smoothness_norm_rmse : float the same RMSE divided by the budget so it
            can be compared across budgets.
        exhaustion_slot : int the first slot at which at least 99.9 percent of
            the budget was spent, or n_slots if it was never reached.
    """
    traffic = np.asarray(traffic, dtype=np.float64)
    n_slots = len(traffic)
    budget = float(budget)

    # Prime the pacer with the plan for this run.
    pacer.reset(budget, n_slots, traffic=traffic)

    spend = np.zeros(n_slots, dtype=np.float64)
    throttle_used = np.zeros(n_slots, dtype=np.float64)
    spent_so_far = 0.0
    exhaustion_slot = n_slots

    for slot in range(n_slots):
        eligible = float(traffic[slot])
        throttle = float(pacer.step(slot, eligible, spent_so_far))
        throttle = min(1.0, max(0.0, throttle))
        throttle_used[slot] = throttle

        desired_spend = throttle * eligible
        remaining = budget - spent_so_far
        slot_spend = min(desired_spend, max(remaining, 0.0))
        spend[slot] = slot_spend
        spent_so_far += slot_spend

        if exhaustion_slot == n_slots and spent_so_far >= 0.999 * budget:
            exhaustion_slot = slot

    cumulative = np.cumsum(spend)
    ideal_spend = _ideal_spend_curve(traffic, budget)
    ideal_cumulative = np.cumsum(ideal_spend)

    budget_utilization = float(spent_so_far / budget) if budget > 0 else 0.0
    diff = cumulative - ideal_cumulative
    smoothness_rmse = float(np.sqrt(np.mean(diff ** 2)))
    smoothness_norm_rmse = float(smoothness_rmse / budget) if budget > 0 else 0.0

    return {
        "spend": spend,
        "cumulative": cumulative,
        "throttle": throttle_used,
        "ideal_spend": ideal_spend,
        "ideal_cumulative": ideal_cumulative,
        "traffic": traffic,
        "budget": budget,
        "budget_utilization": budget_utilization,
        "smoothness_rmse": smoothness_rmse,
        "smoothness_norm_rmse": smoothness_norm_rmse,
        "exhaustion_slot": int(exhaustion_slot),
    }
