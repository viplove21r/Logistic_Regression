# Logistic Regression Placement Predictor

A machine learning classification project that predicts whether a student gets placed based on:

- CGPA
- IQ

This project uses **Logistic Regression** from Scikit-learn and visualizes the **decision boundary** for classification.

---

# 📌 Project Overview

Logistic Regression is a supervised machine learning algorithm used for **binary classification problems**.

In this project:

- `1` → Student gets placed
- `0` → Student does not get placed

The model learns a **decision boundary** that separates the two classes using student CGPA and IQ.

---

# 📂 Dataset

Dataset columns:

| Column | Description |
|---|---|
| cgpa | Student CGPA |
| iq | Student IQ |
| placement | Target variable (0 or 1) |

Example:

| cgpa | iq | placement |
|---|---|---|
| 6.8 | 123 | 1 |
| 5.9 | 106 | 0 |

---

# ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Mlxtend
- Pickle

---

# 🚀 Workflow

## 1. Data Preprocessing

- Removed unnecessary column
- Selected input and output features

```python
X = df.iloc[:,0:2]
y = df.iloc[:,-1]
```

---

## 2. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1
)
```

---

## 3. Feature Scaling

Used `StandardScaler` to normalize the features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Why `transform()` on test data?

The scaler learns:
- mean
- standard deviation

from training data only.

Using `fit_transform()` on test data would create data leakage.

---

## 4. Model Training

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train, y_train)
```

---

## 5. Model Evaluation

```python
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```

### Accuracy Achieved

```python
1.0
```

---

## 6. Decision Boundary Visualization

```python
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train, y_train.values, clf=clf, legend=2)
```

The visualization shows how Logistic Regression separates the two classes using a linear boundary.

---

# 📊 Logistic Regression Intuition

Logistic Regression first computes:

\[
z = \beta_0 + \beta_1x_1 + \beta_2x_2
\]

Then applies the sigmoid function:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

This converts values into probabilities between:

\[
[0,1]
\]

Decision rule:

- Probability ≥ 0.5 → Class 1
- Probability < 0.5 → Class 0

---

# 💾 Model Serialization

The trained model is saved using Pickle.

```python
import pickle

pickle.dump(clf, open("model.pkl", "wb"))
```

This allows the model to be reused later without retraining.

---

# ▶️ How to Run

## Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend
```

---

## Run the notebook/script

```bash
python app.py
```

or open the Jupyter Notebook.

---

# 📈 Future Improvements

- Add more features
- Deploy using Flask/Streamlit
- Hyperparameter tuning
- Cross-validation
- Use larger datasets

---

# 📚 Concepts Covered

- Binary Classification
- Logistic Regression
- Feature Scaling
- Train-Test Split
- Decision Boundary
- Model Evaluation
- Pickle Serialization

---

# 👨‍💻 Author

Viplove Rajora
