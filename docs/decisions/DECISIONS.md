# FARZ — Confirmed Decisions

1. Project name: FARZ | فَرز.
2. Scope: final project for a deep learning course; keep it focused and deliverable.
3. Five independent classes: Glass, Paper, Cardboard, Plastic, Metal.
4. Trash is excluded. No class merging.
5. Current image counts: Glass 501, Paper 594, Cardboard 403, Plastic 482, Metal 410; total 2,390.
6. Original images must not be overwritten or deleted during audits.
7. Do not use the supplied legacy train/validation/test split; create a new leakage-aware split after duplicate/label review.
8. App: Flutter, Android first, Arabic and English.
9. App appearance: Minimal & Modern, Light/Dark/System theme modes.
10. Classification must work fully offline on-device.
11. PyTorch is used for model training and evaluation. ExecuTorch is the planned runtime, subject to a small compatibility proof-of-concept.
12. Google Colab for training; GitHub for source control and build automation. Cloud development is preferred because Flutter is not installed locally.
