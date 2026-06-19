# 🏡 California Home Value Estimator — End-to-End ML Application

A production-ready, end-to-end Machine Learning application that predicts median housing prices in California using geographic and demographic data. This project demonstrates the full ML lifecycle — from experimentation in Jupyter Notebook to deployment as a live, interactive web app on Streamlit Cloud.

---

## 🚀 Project Overview

Many data science projects stop at the notebook stage. This project goes further by operationalising the model into a production-ready system.

It covers the complete pipeline:

* Data ingestion
* Stratified sampling to reduce bias
* Feature engineering
* Model training and tuning
* Deployment via a web interface

The result is a fully functional application where users can input regional data and receive real-time housing price predictions.

---

## ✨ Key Features

* **Robust Preprocessing Pipeline**
  Automated handling of numerical and categorical features (e.g., `ocean_proximity`) using Scikit-Learn pipelines.

* **Optimised Models**
  Fine-tuned using Grid Search and Randomised Search to minimise Root Mean Squared Error (RMSE).

* **Interactive Web App**
  Streamlit interface for real-time predictions based on user inputs.

* **Production-Ready Assets**
  Serialized preprocessing pipeline and trained model for consistent inference.

---

## 🏗️ Machine Learning Pipeline

This project follows a structured and reproducible ML workflow:

1. **Data Ingestion**
   Load and validate the California housing dataset.

2. **Stratified Sampling**
   Use `StratifiedShuffleSplit` on income categories to maintain representative distributions.

3. **Exploratory Data Analysis (EDA)**
   Visualise geographic patterns, distributions, and anomalies.

4. **Feature Engineering**
   Create derived features such as:

   * Rooms per household
   * Population per household

5. **Data Cleaning**
   Handle missing values (e.g., `total_bedrooms`) and outliers.

6. **Categorical Encoding**
   Convert text-based features into numerical format.

7. **Pipeline Construction**
   Bundle preprocessing steps into a reusable `preprocessor.pkl` to prevent data leakage.

8. **Model Training & Tuning**
   Evaluate multiple models and optimise using Grid Search.

9. **Model Serialization**
   Save trained models using `joblib`.

10. **Deployment**
    Serve predictions through a Streamlit web application.

---

##  Tech Stack

* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualisation:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`scikit-learn==1.8.0`)
* **Model Persistence:** Joblib
* **Deployment:** Streamlit

---

##  Project Structure

```text
├── app.py                      # Streamlit application
├── preprocessor.pkl            # Preprocessing pipeline
├── Best_model(Gridsearch).pkl  # Trained model
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https:https://github.com/Jammijuix/linear_reg_california_-housing_project.git
cd linear_reg_california_-housing_project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

##  Deployment & MLOps Insights

###  Separation of Concerns

Early issues (e.g., `AttributeError`) were resolved by ensuring:

* Raw inputs → `preprocessor.transform()`
* Transformed data → model prediction

###  Dependency Management

Strict version pinning (e.g., `scikit-learn==1.8.0`) prevents incompatibility between environments.

###  Model Security

Storing `.pkl` files in public repositories can be risky.
For production systems:

* Use secure object storage
* Or model registries (e.g., Vertex AI)
* Access via authenticated APIs

---

## 🤝 Contact & Collaboration

**Kelvin** — ML & Systems Engineer

* LinkedIn: https://linkedin.com/in/nnamdi-etumnu01
* GitHub: https://github.com/Jammijuix

Open to collaborating on impactful, AI-driven products and systems.

---
