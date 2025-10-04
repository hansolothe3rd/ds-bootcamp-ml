# Data Science Bootcamp: Repo 3 (Days 21–30)

Welcome to **Repo 3** of the 100 Days of Code – Data Science Bootcamp!  
This phase focuses on **Machine Learning (ML) fundamentals**, applying concepts learned in the first two repos to real-world datasets.

---

## 📅 Roadmap

### Day 21 – Introduction to Machine Learning

**Concepts Covered:**
- What is Machine Learning? Supervised vs. Unsupervised.
- The ML workflow: data prep → training → evaluation.
- Introduction to regression with real housing data.

**Practice:**
- Loaded the California Housing dataset (`fetch_california_housing`).
- Split data into training/testing sets.
- Trained and evaluated a Linear Regression model.

**Mini Project: Predict California Housing Prices**
- Built a regression model on the California dataset.
- Evaluated model with MSE and R² score.
- Identified important features driving housing prices.


## Day 22 – Linear Regression Deep Dive (Ames Housing)

**Concepts Covered:**
- Multiple Linear Regression.
- Model assumptions: linearity, independence, homoscedasticity, normality, multicollinearity.
- Metrics: MSE, R² Score.

**Practice:**
- Loaded Ames Housing dataset from OpenML.
- Trained multiple regression using numeric features.
- Evaluated model with MSE and R².
- Visualized residuals and feature importance.

**Mini Project: Predict House Prices (Variation)**
- Built regression model with a subset of 10 features.
- Compared reduced vs full-feature model.
- Plotted actual vs predicted prices.
- Reflected on most important predictors of house prices.


## Day 23 – Logistic Regression (Classification)

**Concepts Covered:**
- Logistic Regression for binary classification.
- Key metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
- Model assumptions: linearity of log-odds, independence, low multicollinearity.

**Practice:**
- Loaded Titanic dataset and preprocessed missing values and categorical features.
- Trained Logistic Regression model.
- Evaluated using accuracy, classification report, ROC-AUC.
- Plotted confusion matrix for visual assessment.

**Mini Project: Titanic Survival Prediction (Variation)**
- Built model using a subset of features.
- Compared performance to full-feature model.
- Visualized predicted probabilities.
- Reflected on most important predictors of survival.


## Day 24 – K-Nearest Neighbors (KNN)

**Concepts Covered:**
- KNN for classification
- Distance metrics and number of neighbors
- Weighted vs unweighted voting

**Practice:**
- Built KNN classifier on Titanic dataset
- Compared accuracy and confusion matrix with different k values

**Mini Project: KNN on Titanic**
- Trained and compared models with k=3 and k=7
- Plotted confusion matrices
- Experimented with distance weighting
- Reflected on the impact of k choice and weighting on accuracy


## Day 25 – Decision Trees & Random Forests

**Concepts Covered:**
- Decision Trees for classification
- Gini impurity & entropy
- Overfitting in trees
- Random Forest ensembles
- Feature importance visualization

**Practice:**
- Implemented a Decision Tree on Titanic dataset
- Visualized the tree structure

**Mini Project: Titanic Survival Prediction**
- Tasked with building Decision Tree & Random Forest
- Compare their accuracy and confusion matrices
- Visualize feature importance
- Reflection on model performance


## Day 26 – Logistic Regression (Multiclass) & Evaluation Metrics

**Concepts Covered:**
- Logistic Regression for classification
- Multiclass strategies (OvR, multinomial)
- Evaluation metrics: accuracy, precision, recall, F1
- Confusion matrices

**Practice:**
- Trained a multinomial Logistic Regression on the Iris dataset
- Evaluated using classification report & confusion matrix

**Mini Project: Multiclass Wine Classification**
- Tasked with training a Logistic Regression on the Wine dataset
- Evaluate with classification report & confusion matrix
- Reflect on which classes were hardest to predict


## Day 27 – Support Vector Machines & Kernel Tricks

**Concepts Covered:**
- SVM for classification
- Linear and non-linear kernels
- Maximum-margin hyperplanes
- Support vectors and regularization

**Practice:**
- Trained SVM with linear kernel on Iris dataset
- Evaluated with classification report and confusion matrix
- Visualized predictions

**Mini Project: SVM Wine Classification**
- Train linear and RBF SVM on Wine dataset
- Compare performance and misclassifications
- Reflect on kernel choice, class performance, and support vectors


## Day 28 – Feature Scaling & PCA

**Concepts Covered:**
- Standardization and normalization
- PCA for dimensionality reduction
- Explained variance and component choice

**Practice:**
- Scaled Iris dataset
- Applied PCA (2D)
- Visualized separability of classes
- Checked explained variance ratio

**Mini Project: PCA on Wine Dataset**
- Scaled features and applied PCA
- Compared Logistic Regression performance on full vs reduced dataset
- Reflected on trade-offs of PCA


### Day 29 – PCA (Dimensionality Reduction)
- Eigenvalues, eigenvectors, explained variance.
- Mini Project: Visualize high-dimensional data.

### Day 30 – Major Project: ML Pipeline
- End-to-end ML workflow: data cleaning, scaling, feature selection, model training, evaluation.
- Suggested Dataset: Wine Quality, Titanic, or a real-world dataset of your choice.
- Deliverable: Full Jupyter notebook + README write-up.

---

## 🛠️ Tools & Libraries
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib` / `seaborn`

---

## 🎯 Major Project (Day 30)
At the end of Repo 3, you’ll build a **complete ML pipeline**:
- Load & clean data.
- Explore & preprocess.
- Train multiple ML models.
- Evaluate & compare performance.
- Present insights & recommendations.

---
