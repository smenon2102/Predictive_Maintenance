---
title: Predictive Maintenance App
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Predictive Maintenance App

This application predicts whether an engine is healthy or requires maintenance based on sensor readings.

## Input Features
- Engine RPM
- Lubricating Oil Pressure
- Fuel Pressure
- Coolant Pressure
- Lubricating Oil Temperature
- Coolant Temperature

## Output
- Engine is Healthy
- Engine Requires Maintenance

The app loads the final AdaBoost model directly from the Hugging Face Model Hub and performs real-time inference using user-provided sensor values.
