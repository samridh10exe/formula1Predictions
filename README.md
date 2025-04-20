# Formula One Race Time Prediction: 2025 Saudi Arabian Grand Prix



This project uses machine learning techniques to predict race times for the **2025 Saudi Arabian GP** at the Jeddah Corniche Circuit. By using real-world qualifying data and engineered lap features, I explored two different regression models to predict final race times.

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

### Prediction Output

```
Predicted Results for 2025 Saudi Arabian GP (complex Model)
-------------------------------------------------------------
0         Charles Leclerc              91.469759
1          Lewis Hamilton              91.469759
2           Oscar Piastri              91.477881
3          Max Verstappen              91.477881
4          George Russell              91.477881
5   Andrea Kimi Antonelli              91.477881
6            Yuki Tsunoda              91.477881
7            Pierre Gasly              91.477881
8          Oliver Bearman              91.477881
9            Lando Norris              91.477881
10            Liam Lawson              91.477881
11           Isack Hadjar              91.477881
12      Gabriel Bortoleto              91.477881
13           Esteban Ocon              91.477881
14        Nico Hülkenberg              91.477881
15            Jack Doohan              91.477881
16        Fernando Alonso              91.481554
17           Lance Stroll              91.481554
18        Alexander Albon              91.483384
19       Carlos Sainz Jr.              91.483384
```

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

### Prediction Output

```
Predicted Results for 2025 Saudi Arabian GP (Simple Model)
-----------------------------------------------------------
0	Charles Leclerc	90.125458
1	Kimi Antonelli	90.125458
2	Max Verstappen	90.167093
3	Oscar Piastri	90.385852
4	Yuki Tsunoda	91.425216
5	Alexander Albon	91.707594
6	Nico Hulkenberg	91.879882
7	Jack Doohan	92.146129
8	Lando Norris	92.391439
9	Carlos Sainz	92.527491
10	Gabriel Bortoleto	92.656598
11	Esteban Ocon	92.656598
12	George Russell	92.716397
13	Lewis Hamilton	92.958050
14	Liam Lawson	93.664730
15	Pierre Gasly	93.995143
16	Isack Hadjar	93.995143
17	Lance Stroll	94.090833
18	Fernando Alonso	94.178002
19	Oliver Bearman	97.754132
```

---

## Key Engineering Techniques

- **One-Hot Encoding:** Applied to `Compound` and `Team` columns for GBR model
- **Data Alignment:** Ensured feature columns matched during inference
- **Custom Mapping:** Translated driver names to unique codes for merging qualifying and lap data
- **Gradient Boosting:** Tuned with `n_estimators=200`, `learning_rate=0.1`

---

## Evaluation Metrics

| Model Type            | MAE (seconds) |
|-----------------------|---------------|
| Feature-Rich Model    | 13.475        |
| Qualifying-Only Model | 1.320         |

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
