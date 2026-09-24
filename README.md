# Alarm Burden in Embedded Arrhythmia Detection

Implementation and measurements behind two papers on false alarms in
microcontroller-class arrhythmia detection: the conversion from per-beat metrics
to alarms per 24 h, and the episode-gated verification module that reduces them.

## Pipeline

| Directory | Stage |
|---|---|
| `01_data/` | build MIT-BIH DS1/DS2 and the 437.5 h NSRDB corpus |
| `02_detectors/` | three reimplemented detectors, trained and calibrated |
| `03_conversion/` | per-beat metrics to alarms per 24 h (episode rule, refractory window) |
| `04_module/` | the episode-gated verification module |
| `05_evaluation/` | alarm rates, matched-sensitivity comparison, error overlap |
| `Data/` | provenance only; no corpus is redistributed |
| `06_figures/` | scripts that generate every figure in the papers |
| `results/` | raw run outputs quoted in the papers |

## Headline results

Trained on MIT-BIH DS1, evaluated patient-disjoint on DS2 and on 437.5 h of
normal sinus rhythm (1,806,778 beats), where every alarm is false by
construction.

| Detector | Params | DS2 Se/PPV | FA/24 h k=1 | FA/24 h k=6 |
|---|---:|---|---:|---:|
| Busia 2024 | 6,353 | 42.6 / 98.0 | 70.27 | 0.11 |
| Farag 2023 | 1,349 | 20.2 / 86.3 | 110.15 | 2.47 |
| ArrythML 2026 | 166,868 | 0.1 / 5.0 | 201.88 | 145.87 |

The 0.11 figure is not a usable operating point: at k=6 that detector attains
zero episode sensitivity on a partition containing 103 annotated ventricular
runs of length six or greater. Reporting the alarm rate together with the
episode sensitivity retained at the same operating point is the central
methodological claim.

The episode-gated module retains 64% episode sensitivity on DS2 with no
unmatched alarm, against 43% at 6.5 false alarms per 24 h for temporal gating
alone. Activating the verifier once per candidate episode rather than once per
positive beat raises sensitivity from 7% to 26% at k=2 while reducing
activations on DS2 by a factor of 47.

## Reproducing

    python3 -m venv .venv
    ./.venv/bin/pip install numpy scipy scikit-learn torch wfdb matplotlib

    ./.venv/bin/python 01_data/fetch.py          # MIT-BIH DS1, ~10 min
    ./.venv/bin/python 01_data/fetch_ds2.py
    ./.venv/bin/python 01_data/build2.py nsrdb   # 437.5 h NSRDB, ~60 min
    ./.venv/bin/python 02_detectors/train2.py
    ./.venv/bin/python 02_detectors/resume.py    # autoencoder and calibration
    ./.venv/bin/python 05_evaluation/final_eval.py
    ./.venv/bin/python 04_module/episode_gate.py
    ./.venv/bin/python 05_evaluation/module_cost.py
    ./.venv/bin/python 05_evaluation/corr.py

Thresholds are calibrated on a held-out 20% of DS1 records so that each detector
flags 1% of normal beats, placing the three at a common operating point.

## Limitations recorded in the papers

Episode counts on DS2 are small: 145 annotated ventricular runs at k=2 and 19 at
k=3, and two at k=5, so estimates at clinically relevant episode lengths rest on
few events. Peak sensitivity across all configurations is 64%, a property of the
primary detector under inter-patient evaluation rather than of the module. The
Farag reimplementation attains 20.2% sensitivity against 98.18% reported; this
is a property of the reimplementation and should not be read as a statement
about that method.

## Citation

See CITATION.cff.

## License

MIT
