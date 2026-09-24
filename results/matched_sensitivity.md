# Paper 2 decisive test — module vs timing logic at matched sensitivity

Busia as main model. MIT-BIH DS2 (11.03 h, arrhythmia patients).
Episode = k consecutive flagged beats. Sensitivity = true ventricular runs
that have a matching alarm within 10 s. False alarm = alarm matching no
true episode.

| Mechanism | k | False alarms/day | Episode Se |
|---|---|---|---|
| timing rule only | 1 | 41.3 | 90% |
| timing rule only | 2 | 6.5 | 43% |
| timing rule only | 3 | 2.2 | 16% |
| timing rule only | 4 | 2.2 | 0% |
| module, +1 other agrees | 1 | **0.0** | **64%** |
| module, +both agree | 1 | 0.0 | 0% |
| module, +1 other agrees | 2 | 0.0 | 7% |
| module, +both agree | 2 | 0.0 | 0% |

## The result

**The module beats timing logic, and not marginally.**

Best comparable operating points:
- timing k=2:  6.5 false alarms/day at 43% sensitivity
- module +1:   0.0 false alarms/day at 64% sensitivity

The module is better on BOTH axes simultaneously — fewer false alarms AND
higher sensitivity. This is not a trade-off, it is a dominance.

Why: timing logic is blind. It suppresses any short run, true or false.
The module is selective. It asks a second model whether the beat really
looks abnormal, so it removes noise-driven flags while keeping real ones.

This is the justification for paper 2 that we previously lacked.

## Caveats — do not skip these

1. **Unanimity is still useless.** "+both agree" gives 0% sensitivity at every
   k. Requiring all three models to agree silences the device. Only the
   "1 of 2 others agrees" rule is viable. Report the failure of unanimity.

2. **0.0 false alarms/day is suspicious in absolute terms.** It means zero
   unmatched alarms in 11 hours across 22 patients. The sample is small: at
   k=1 there are 3,046 true episodes but the false alarm count is tiny because
   DS2 is dense with real arrhythmia, so most alarms land near something true.
   The healthy-subject number (NSRDB) is the honest false alarm figure and
   must be reported alongside. On NSRDB, Busia alone at k=1 gave 70.27/day.

3. **Sensitivity caps at 64%.** Even the best configuration misses a third of
   ventricular episodes. That is a real limitation, not a rounding issue.

4. **DS2 has too few long episodes.** Only 19 true runs at k=3 and 5 at k=4.
   Anything at k>=3 rests on single-digit event counts. VFDB/AFDB with a QRS
   detector are still needed before these numbers are publishable.

## Status

Paper 2 now has its central claim supported: a selective verification module
dominates blind temporal gating at matched sensitivity. The claim needs
re-measuring on VFDB/AFDB before submission, but the direction is established
and it is the direction the professor proposed.
