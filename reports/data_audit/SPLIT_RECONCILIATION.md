# Experiment 01 split reconciliation

The uploaded Colab data_split.csv is the canonical split used to train and test the first FARZ model.

| Split | Images |
|---|---:|
| Train | 1,705 |
| Validation | 339 |
| Test | 340 |
| Total | 2,384 |

Manifest SHA-256:

`9afb5036e95058134d5491e51808cb5aa952762e55f7848f10203bdba229db34`

Verification performed against the original resized dataset:
- all 2,384 relative paths exist;
- all image SHA-256 hashes match the manifest;
- all 2,384 SHA-256 values are unique;
- 2,360 unique perceptual groups;
- no group crosses train, validation, or test.

A prior 1,702 / 341 / 341 summary came from regenerating the split with the seven-fold splitter. It was not the manifest used in the completed Colab run. The uploaded manifest is preserved in the FARZ project bundle as data/manifests/data_split_run_01.csv. Use that exact file to reproduce Experiment 01; use a separately named split for future experiments.
