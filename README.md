﻿# Satellite_anomaly_detect

## Project Overview
This project focuses on detecting anomalies in satellite telemetry data. Sudden deviations in system behavior can indicate faults or abnormal conditions in satellites, and timely detection is crucial. This model aims to identify such irregular patterns using machine learning techniques.

## Dataset
The dataset was sourced from this repository:  
[Satellite Telemetry Anomaly Detection – Data Source](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/Data)

It contains telemetry data collected from various satellite sensors. The data includes both normal and potentially anomalous readings.

## Tech Stack
- Python
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib

## Methodology
The Isolation Forest algorithm is used to identify anomalies in the satellite telemetry data. This unsupervised learning method isolates anomalies based on the principle that anomalies are few and different.

Results are visualized using graphs, showing clear separations between normal and anomalous data points.

## Key Results
- Anomalies were successfully detected and highlighted using visual plots.
- The model demonstrated the capability to flag unusual behavior without requiring labeled data.
- While no specific accuracy score is reported, visual analysis confirmed the effectiveness of the method.

## Future Work
- Experimenting with other anomaly detection models such as Local Outlier Factor, Autoencoders, or One-Class SVM.
- Adding a comparative study of model performance.
- Incorporating labeled anomaly cases (if available) for supervised learning enhancements.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/satellite-anomaly-detection.git
   cd satellite-anomaly-detection
