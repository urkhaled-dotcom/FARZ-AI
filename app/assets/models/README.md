# App model assets

The verified ExecuTorch model will be placed here:

- farz_model.pte — XNNPACK ExecuTorch program exported from the Experiment 01 checkpoint.
- farz_model_metadata.json — input preprocessing and class order.

The model is not included yet. Run the FARZ ExecuTorch Export Colab notebook and confirm the balanced test parity gate (30/30 top-1 predictions), then copy both files here. Until the PTE exists, the app intentionally reports that its model is unavailable and disables classification.
