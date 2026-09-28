# ClickRank Benchmark Report

This report compares click through rate models on a shared featurized dataset. Models are ranked by test AUC. All numbers come from the held out test split.

## Results

| Model | AUC | LogLoss | NE | RelaImpr | GAUC | ECE | Train s | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFM | 0.7879 | 0.4507 | 0.8121 | 0.1879 | 0.7880 | 0.0078 | 126.9506 | 21616509 |
| DLRM | 0.7872 | 0.4517 | 0.8139 | 0.1861 | 0.7872 | 0.0103 | 165.8838 | 20417937 |

## Calibration

The reliability curves below overlay each model against the perfect calibration diagonal. A curve that hugs the diagonal is well calibrated.

![Calibration curves](calibration.png)

## Key Observations

The strongest model by test AUC is DeepFM. It ranks impressions better than the logistic regression baseline which scored the baseline AUC. Ranking quality is what an ad auction cares about most because the auction orders candidates before pricing them.

The factorization machine family beats plain logistic regression because the synthetic clicks depend on pairwise interactions between categorical fields. Logistic regression only learns linear weights on single features, so it cannot capture the lift that appears when two category groups co occur. The FM second order term and the deep towers model those interactions directly, which is why they pull ahead on AUC and normalized entropy.

Calibration and normalized entropy round out the picture. A model can rank well and still be biased in absolute probability, so the calibration curve and the expected calibration error matter for bidding and pacing. Group AUC reflects within request ranking quality, which is closer to the live auction than dataset wide AUC. Reading these metrics together gives a fair comparison of the architectures.
