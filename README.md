# Formula One Race Prediction



My first version of this project uses machine learning to predict race times for the **2025 Saudi Arabian GP** at the Jeddah Corniche Circuit. By using real-world qualifying data and engineered lap features, I explored two different regression models to predict final race times. as of 4/19/2025

---

## Project Objective

- Predict **race completion times (in seconds)** for all 20 drivers using their qualifying results and other available telemetry.
- Experiment with two modeling approaches:  
  - A **feature-rich Gradient Boosting Regressor (GBR)** using historical lap and weather data.
  - A **simple model** trained only on qualifying times.

---

## Technical Stack

- **Python 3.11**
- **pandas**, **scikit-learn**, **NumPy**
- **GradientBoostingRegressor** for both complex and simple models
- **FastF1** for lap time extraction (historical training data)
- **Jupyter Notebook** for exploration and evaluation

---

## Dataset Overview

- **Training Data:** Historical lap time data from the 2024 Saudi Arabian GP (`laps_2024`)
- **Test Input:** Simulated 2025 qualifying session (`qualifying_2025`)
- Each entry includes:
  - Driver name
  - Team
  - Tyre compound
  - Environmental conditions: humidity, pressure, wind speed, etc.
  - Track temperature & air temperature

---

## Model 1: Feature-Rich GBR Model

Trained on extensive telemetry and environmental data with one-hot encoding applied to categorical features like team and tyre.

- **MAE:** ~13.47 seconds
- **Feature count after encoding:** ~50+


All drivers were aligned and re-indexed to match model input features using:
```python
input_data = pd.get_dummies(qualifying_2025)
input_data = input_data.reindex(columns=features.columns, fill_value=0)
```

---

## Model 2: Qualifying-Only Model

A simpler pipeline using just the driver's **qualifying time** as the only feature.

- **MAE:** ~1.32 seconds
- **Model:** GradientBoostingRegressor with 200 estimators


## Key Engineering Techniques

- **One-Hot Encoding:** Applied to `Compound` and `Team` columns for GBR model
- **Data Alignment:** Ensured feature columns matched during inference
- **Custom Mapping:** Translated driver names to unique codes for merging qualifying and lap data
- **Gradient Boosting:** Tuned with `n_estimators=200`, `learning_rate=0.1`
- **Multi-Model Comparison:** Evaluated Ridge Regression, Random Forest, XGBoost, and LightGBM against the baseline

The surprising effectiveness of Ridge Regression (with a mere 0.0048s MAE) suggests that while F1 racing is complex, the relationship between qualifying and race times may have stronger linear components than initially hypothesized.
Also evident by the result of last 3 pole positions vs race winners.
Out of the four completed races in the 2025 Formula 1 season, three pole-sitters have won: Lando Norris in Australia, Max Verstappen in Japan, and Oscar Piastri in Bahrain


---

## Model Comparison & Evaluation Metrics

Initial models:
| Model Type            | MAE (seconds) |
|-----------------------|---------------|
| Feature-Rich Model    | 13.475        |
| Qualifying-Only Model | 1.320         |

Extended model comparison:
| Model             | MAE (seconds) | R²       | Training Time (s) |
|-------------------|---------------|----------|-------------------|
| Ridge Regression  | 0.0048        | 0.999982 | 0.03              |
| Gradient Boosting | 0.0548        | 0.997123 | 0.70              |
| XGBoost           | 0.0698        | 0.993462 | 0.28              |
| Random Forest     | 0.0832        | 0.990834 | 3.43              |
| LightGBM          | 0.0903        | 0.988556 | 0.18              |

The Ridge Regression model surprisingly outperformed other algorithms with the lowest MAE of just 0.0048 seconds, despite being a simpler linear model. This suggests that for this particular dataset and prediction task, the relationship between features and lap times may be more linear than initially expected.

*For detailed race predictions from each model (converted to full race times), see the [predictions.md](predictions.md) file.*


---

## Future Improvements

- Integrate **pit stop strategies** and **tire degradation curves**
- Add **real-time telemetry and sector analysis**
- Use **XGBoost or LightGBM** for model comparison
- Predict **race position rankings** in addition to time

## Work in Progress: Sentiment Analysis

Currently developing a sentiment analysis component to analyze:
- Driver radio communications during races
- Social media reaction to race predictions and results
- Team press release, pre and post-race interviews


---

## Notebook Access

This repository includes:
- `2025_Saudi_Arabiangp.ipynb`: Main notebook with full pipeline, training, prediction, and result visualization

---

## License

MIT License — open to use with attribution.
