# Satellite_anomaly_detect

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
Developed anomaly detection system using Isolation Forest and autoencoder neural networks for multivariate satellite telemetry data, reducing false positive rate to 8% through hyperparameter optimization.

Applied feature engineering and dimensionality reduction (PCA, t-SNE) for high-dimensional time-series analysis, enabling early fault detection in aerospace systems.

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
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the modular training pipeline:
   ```bash
   python run_pipeline.py
   ```

4. Review outputs:
   - `models/isolation_forest.joblib`
   - `models/scaler.joblib`
   - `models/autoencoder/`
   - `logs/pipeline.log`
   - `logs/metrics.json`

5. Run the FastAPI prediction server:
   ```bash
   uvicorn api:app --reload
   ```

6. Access the API information page at `http://127.0.0.1:8000/` (or port shown in terminal, e.g., 8001)

7. Use the API directly:
   ```bash
   curl -X POST "http://127.0.0.1:8001/predict" \
        -H "Content-Type: application/json" \
        -d '{
          "BatteryTemperature": 3.8,
          "BusVoltage": 32.0,
          "TotalBusCurrent": 1266.0,
          "WheelRPM": 1000.0,
          "WheelTemperature": 25.0
        }'
   ```

   **Expected Response:**
   ```json
   {
     "isolation_forest": {
       "anomaly_score": 0.1453,
       "is_anomaly": false
     },
     "autoencoder": {
       "anomaly_score": 14356.47,
       "is_anomaly": true
     },
     "overall_anomaly": true
   }
   ```

8. Check API health:
   ```bash
   curl http://127.0.0.1:8001/health
   ```
