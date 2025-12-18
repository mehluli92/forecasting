# 🛒 M5 Supply Chain Demand Forecasting

**A Non-Parametric Framework for Large-Scale Retail Prediction**

### 📋 Executive Summary
This project builds a robust end-to-end forecasting pipeline to predict unit sales for Walmart retail goods over a 28-day horizon. Using a dataset of over 30,000 unique item-store combinations (M5 Competition data), the solution employs a hybrid strategy leveraging **Facebook Prophet** for feature engineering and **LightGBM** for final prediction.

The final model achieved a validation **RMSE of 1.93**, successfully identifying distinct seasonal patterns across different departments (Foods, Hobbies, Household).

---

### 🛠 Tech Stack & Tools
* **Languages:** Python 3.9+
* **Machine Learning:** LightGBM, TensorFlow/Keras (LSTM), Scikit-Learn
* **Time Series Analysis:** Facebook Prophet
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

### 🧠 Methodology & Approach

#### 1. Advanced Feature Engineering (The "Secret Sauce")
Instead of using raw time-series data, I used **Facebook Prophet** to extract latent seasonal signals.
* **Innovation:** Rather than a single global trend, I trained separate Prophet models for each category (`FOODS`, `HOBBIES`, `HOUSEHOLD`).
* **Signal Extraction:** Extracted weekly patterns, yearly seasonality, and holiday effects (Super Bowl, Black Friday) specific to each department.
* These components were injected as features into the downstream supervised learning models.

#### 2. Model Selection & Experimentation
I experimented with two primary architectures:
1.  **LightGBM (Gradient Boosting):** A tree-based model optimized for tabular features and categorical data.
2.  **LSTM (Deep Learning):** A Recurrent Neural Network designed to capture sequential dependencies.

#### 3. The Hybrid Ensemble Experiment
I attempted a weighted ensemble of `(0.9 * LightGBM) + (0.1 * LSTM)`.
* **Hypothesis:** Combining deep learning and tree-based models would reduce variance.
* **Result:** The ensemble (RMSE 1.97) underperformed compared to the standalone LightGBM model (RMSE 1.93).
* **Conclusion:** The feature engineering performed via Prophet provided such strong signal clarity that the LightGBM model outperformed the complex deep learning approach. **I prioritized the simpler, more accurate model for the final solution.**

---

### 📂 Repository Structure

| File | Description |
| :--- | :--- |
| **`enhanced_prophet.ipynb`** | **Feature Engineering Pipeline.** Contains the logic for category-specific trend extraction using Prophet. |
| **`main.ipynb`** | **Training & Evaluation.** Preprocessing, LightGBM model training, LSTM construction, and final error analysis. |
| **`lgbm_predictions.csv`** | Final forecast output for the validation period. |

---

### 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mehluli92/forecasting.git](https://github.com/mehluli92/forecasting.git)
    cd forecasting
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy lightgbm prophet tensorflow matplotlib
    ```

3.  **Run the Notebooks:**
    Start with `enhanced_prophet.ipynb` to generate features, then run `main.ipynb` for the final forecast.

---

### 📈 Key Results
* **Baseline (LightGBM w/ Prophet Features):** 1.9307 RMSE ✅ *(Winner)*
* **Ensemble (Hybrid):** 1.9784 RMSE

---

### 👤 Author
**Mehluli Nokwara**
* *Focus: Deep Learning & Time Series Forecasting*
* *Course Context: Deep Learning (Dr. Ramesh)*
