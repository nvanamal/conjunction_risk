# conjunction_risk
Leakage aware machine learning for early prediction of final conjunction risk
PROJECT 2A — CONJUNCTION-RISK FORECASTING

File
----
project2a_conjunction_risk_full_analysis.py

Purpose
-------
This single-file pipeline performs leakage-aware, event-level forecasting of
the final reported conjunction-risk estimate from sequential Conjunction Data
Messages (CDMs). It includes:

- fixed decision horizons of 1, 2, 3, 5, and 6 days before TCA;
- snapshot-only and sequence-trend feature comparisons;
- single-stage Extra Trees regression;
- a floor/non-floor two-stage sensitivity analysis;
- classification and regression metrics; and
- CSV outputs, a horizon figure, and run metadata.

Requirements
------------
Python 3.10 or newer is recommended.

Install the required packages:

    py -3.11 -m venv .venv
    .venv\Scripts\activate
    pip install numpy pandas scikit-learn matplotlib

Quick synthetic smoke test
--------------------------

    python project2a_conjunction_risk_full_analysis.py --demo --quick

The demo is synthetic and must not be reported as a research result.

Real ESA data
-------------
Place the following files together in one folder:

- train_data.csv
- test_data.csv
- test_data_private.csv

Then run:

    python project2a_conjunction_risk_full_analysis.py --data-dir PATH_TO_FOLDER

The private-label file is expected to contain event_id and true_risk. Training
and test CDM files must contain at least event_id, time_to_tca, and risk, plus
the predictor columns used by the script.

Outputs
-------
By default, results are written to project2a_outputs/:

- horizon_results.csv
- classification_results.csv
- availability.csv
- official_test_predictions.csv
- run_metadata.json
- figures/horizon_mae.png

Data and licensing note
-----------------------
The ESA Collision Avoidance Challenge dataset is not included. Obtain it from
its authorized source and review its license before redistributing any data.
Do not upload private, restricted, employer-owned, or credential-containing
files to a public GitHub repository.

Scientific caution
------------------
The target is the final reported risk estimate, not an observed collision
outcome. This code is a research workflow and is not an operational warning,
collision-avoidance, or maneuver-recommendation system.

The data-driven event-state representation should not be described as a
validated, physics-based operational digital twin. Basilisk/HCW simulation and
prospective live-CDM validation remain future work unless independently run.

Author
------
Nagendrababu Vanamala, Ph.D.
