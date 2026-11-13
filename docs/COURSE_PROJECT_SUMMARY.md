# FARZ — Deep Learning Course Final Project

## Problem
People may find it difficult to identify the material of common recyclable waste items. FARZ demonstrates how an image-classification model can help identify one item at a time.

## Objective
Train and evaluate a five-class image classifier using PyTorch, then package the trained model into an Android app that performs inference locally without an internet connection.

## Classes
Glass, Paper, Cardboard, Plastic, Metal. Trash is excluded. No classes are merged.

## Scope
- One waste item per image.
- Android-first app.
- Arabic and English UI.
- Light, Dark, and System appearance modes.
- On-device inference, no inference server.
- Basic history and sorting guidance stored/displayed locally.

## Evaluation
Report overall accuracy, macro precision/recall/F1, per-class metrics, confusion matrix, model file size, and on-device inference latency. Include representative failure cases, particularly Paper vs Cardboard and Glass vs Plastic.

## Out of scope
Multi-object detection, cloud accounts, backend service, web dashboard, and iOS delivery.
