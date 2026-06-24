# DevelopersHub Corporation – AI/ML Engineering Internship

**Intern Name:** Bano
**Due Date:** 26th June, 2026

## Tasks Completed

### Task 1: Dataset Visualization
- **Objective:** Explore and visualize the Iris dataset
- **Dataset:** Iris Dataset (seaborn)
- **Key Steps:** Loaded and inspected data with pandas, visualized feature relationships using scatter plots, histograms, and box plots
- **Key Findings:** Setosa species is clearly separable from the other two species based on petal length/width

### Task 2: Heart Disease Prediction
- **Objective:** Predict heart disease risk from health data
- **Dataset:** Heart Disease UCI Dataset (Kaggle)
- **Model:** Logistic Regression and Decision Tree
- **Evaluation:** Accuracy, ROC-AUC, and confusion matrix
- **Key Finding:** Chest pain type and maximum heart rate were among the top predictors of heart disease risk

### Task 3: Mental Health Support Chatbot (Fine-Tuned)
- **Objective:** Build a chatbot that provides supportive, empathetic responses for stress, anxiety, and emotional wellness
- **Model Base:** DistilGPT2
- **Dataset:** EmpatheticDialogues (Facebook AI)
- **Approach:** Fine-tuned DistilGPT2 using Hugging Face's Trainer API on a subset of empathetic dialogue pairs; built a command-line interface to test conversational responses
- **Key Finding:** The fine-tuned model could generate context-aware, emotionally supportive replies, though response coherence is limited by the small model size

### Task 4: House Price Prediction
- **Objective:** Predict house prices using property features such as size, number of rooms, age, and distance to city
- **Dataset:** House Price Prediction Dataset (Kaggle)
- **Models:** Linear Regression and Gradient Boosting Regressor
- **Evaluation:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
- **Key Finding:** Gradient Boosting outperformed Linear Regression by capturing non-linear relationships between property features and price

## Tools & Libraries Used
- Python, pandas, NumPy
- scikit-learn
- Hugging Face Transformers & Datasets
- matplotlib, seaborn
- Google Colab (T4 GPU) for model fine-tuning
