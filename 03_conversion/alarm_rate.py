"""Conversion from per-beat operating points to false alarms per 24 h.

Implements Section III of the manuscript. Three stages, each a free parameter
that published work leaves implicit:

  1. prevalence correction   (Sec. III-A, Eq. 1)
  2. episode rule            (Sec. III-B)  k consecutive positives, gap <= tau_g
  3. refractory window       (Sec. III-C)  tau_r after each issued alarm

Every configuration is reported as the pair (alarms per 24 h, episode
sensitivity); see Sec. III-D. A rate alone is uninterpretable because any
suppressive mechanism drives it to zero by suppressing every alarm.
"""

import numpy as np

FS_DEFAULT   = 360.0     # Hz, MIT-BIH sampling rate
TAU_GAP      = 1.2       # s, maximum inter-beat gap inside one episode
TAU_REFR     = 300.0     # s, refractory window (Sec. III-C)
MATCH_TOL    = 10.0      # s, onset tolerance for calling an alarm true
ASSURE_BOUND = 0.29      # false positive alarms per subject-day, NCT03887052


# --- Stage 1: prevalence correction (Sec. III-A) --------------------------

def precision_from_f1_recall(f1, recall):
    """Recover positive predictivity where a work reports F1 and recall only."""
    denom = 2.0 * recall - f1
    if denom <= 0:
        raise ValueError("inconsistent F1 and recall")
    return f1 * recall / denom


def fpr_from_operating_point(precision, sensitivity, corpus_prevalence):
    """Equation (1). Returns the prevalence-invariant false positive rate."""
    if not 0.0 < precision <= 1.0:
        raise ValueError("precision out of range")
    num = sensitivity * corpus_prevalence * (1.0 - precision)
    den = precision * (1.0 - corpus_prevalence)
    return num / den


# --- Stage 2 and 3: episodes and refractory window (Sec. III-B, III-C) ----

def episodes(flag, t, k, tau_gap=TAU_GAP, tau_refr=TAU_REFR):
    """Group a beat-level decision sequence into issued alarm onsets.

    flag : 0/1 array of per-beat decisions, one record, time-ordered
    t    : beat times in seconds, same length as flag
    k    : minimum run length for a candidate episode
    """
    out, n, i, last = [], len(flag), 0, -np.inf
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1] and (t[j + 1] - t[j]) <= tau_gap:
                j += 1
            if j - i + 1 >= k and t[i] - last >= tau_refr:
                out.append(t[i])
                last = t[i]
            i = j + 1
        else:
            i += 1
    return out


def score(alarm_onsets, reference_onsets, tol=MATCH_TOL):
    """Sec. III-D. Returns (false alarm count, episode sensitivity).

    An alarm is false when no reference episode begins within tol of it; a
    reference episode is detected when some alarm falls within tol of its onset.
    Counted separately so that one alarm spanning two references inflates
    neither quantity.
    """
    ref = np.sort(np.asarray(reference_onsets, dtype=float))
    det = 0
    for r in ref:
        if any(abs(a - r) < tol for a in alarm_onsets):
            det += 1
    false = sum(1 for a in alarm_onsets
                if not any(abs(a - r) < tol for r in ref))
    return false, det / max(len(ref), 1)


def per_24h(count, hours):
    return count * 24.0 / hours
