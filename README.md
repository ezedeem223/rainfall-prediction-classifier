# Rainfall Prediction in Australia

This project implements a machine learning model to predict whether it will rain the next day in Australia. The model is built and trained using a dataset published on Kaggle.

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg )](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package )
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg )](https://www.python.org/downloads/ )
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg )](https://opensource.org/licenses/MIT )

---

## 🎯 Project Goal

The main goal is to build an accurate classifier that predicts the `RainTomorrow` variable based on other weather features available in the dataset. Several algorithms were explored and evaluated to achieve the best possible performance.

---

## 📂 Project Structure

rainfall-prediction-classifier/
│
├── rainfall prediction classifier.ipynb    
├── README.md                               
└── requirements.txt                        

---

## 🛠️ Tech Stack

*   **Programming Language:** Python 3.9
*   **Data Analysis & Processing:** Pandas, NumPy
*   **Data Visualization:** Matplotlib, Seaborn
*   **Machine Learning Models:** Scikit-learn (Logistic Regression, RandomForestClassifier), XGBoost (XGBClassifier)

---

## 📋 Project Workflow

1.  **Exploratory Data Analysis (EDA):**
    *   Understand data dimensions and variable types (numerical and categorical).
    *   Handle Missing Values.
    *   Analyze the distribution of the target variables (`RainToday`, `RainTomorrow`).
    *   Study the correlation between features using Heatmaps.

2.  **Data Preprocessing:**
    *   Separate numerical and categorical features.
    *   Handle outliers in numerical features.
    *   Encode categorical features into numerical format using One-Hot Encoding.
    *   Scale numerical features using `MinMaxScaler`.

3.  **Model Building & Training:**
    *   Split the data into training and testing sets (`train_test_split`).
    *   Train three different models:
        *   Logistic Regression
        *   Random Forest Classifier
        *   XGBoost Classifier

4.  **Model Evaluation:**
    *   Evaluate the accuracy of each model on the test set.
    *   Compare model performance to select the best one.

---

## 📊 Results

The performance of the three models was compared based on their accuracy on the test data:

| Model                        | Train Set Accuracy | Test Set Accuracy |
| ---------------------------- | ------------------ | ----------------- |
| **Logistic Regression**      | 84.03%             | 84.24%            |
| **Random Forest Classifier** | 99.99%             | 84.28%            |
| **XGBoost Classifier**       | 87.11%             | **85.19%**        |

**Conclusion:** The **XGBoost Classifier** demonstrated the best performance on the test set, making it the recommended model for this problem.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ezedeem223/rainfall-prediction-classifier.git
    cd rainfall-prediction-classifier
    ```

2.  **(Optional but recommended ) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    *   *For best results, first generate a `requirements.txt` file by running `pip freeze > requirements.txt` in your local environment.*
    *   *Then, anyone can install the dependencies using:*
        ```bash
        pip install -r requirements.txt
        ```

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook "rainfall prediction classifier.ipynb"
    ```
