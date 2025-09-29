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


### Day 25 – Decision Trees
- Gini vs. Entropy, tree depth, overfitting.
- Visualization of decision boundaries.
- Mini Project: Build a decision tree classifier.

### Day 26 – Random Forests
- Ensemble methods, bagging, feature importance.
- Compare single trees vs. forests.
- Mini Project: Predict diabetes outcomes.

### Day 27 – Support Vector Machines (SVM)
- Margin maximization, kernels, soft margins.
- Practice with linear + RBF kernels.
- Mini Project: Handwritten digit recognition (MNIST subset).

### Day 28 – Clustering (Unsupervised Learning)
- k-Means, silhouette score, elbow method.
- Mini Project: Customer segmentation.

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
