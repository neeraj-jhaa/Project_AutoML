#📩 AutoML for Spam Detection
#🎯 Objective

Use AutoML and Hyperparameter Tuning tools to automate the model selection and optimization process for spam detection.

#📖 Theory

AutoML automates key parts of the machine learning pipeline, including:

🔹 Feature extraction

🔹 Model selection

🔹 Model evaluation

In this project:

Multiple classification algorithms were systematically compared using cross-validation:

⚡ Logistic Regression

⚡ Multinomial Naive Bayes

⚡ Linear Support Vector Classifier (SVC)

⚡ Random Forest

A TF-IDF Vectorizer was applied to convert SMS messages into numerical features while:

🧹 Removing stopwords

📝 Considering unigrams and bigrams

➡️ This step was essential because text-based data requires strong feature representation for ML models to perform well.

⚙️ Methodology
🔹 1. Model Evaluation

Metric: F1-score (balances precision & recall)

Reason: Works well for imbalanced datasets like spam detection

✅ Result: Linear SVC outperformed others as the strongest baseline

🔹 2. Hyperparameter Tuning

Framework: Optuna (state-of-the-art optimization tool)

Parameters tuned:

📌 N-gram range (for TF-IDF)

📌 Minimum document frequency (min_df)

📌 Regularization parameter (C) for SVC

➡️ Optuna explored the hyperparameter space efficiently and selected the best configuration to maximize F1-score.

🔹 3. Final Model Training

Trained the optimized SVC model on the training set

Evaluated on the test set

📊 Results
📌 Metric	✅ Ham (Non-Spam)	🚨 Spam
Precision	0.99	0.96
Recall	0.99	0.91
F1-score	0.99	0.94

🏆 Overall Accuracy: 98%

🏆 Weighted F1-score: 0.98

✨ The final tuned model successfully balanced:

Minimizing false positives (ham misclassified as spam)

Minimizing false negatives (missed spam)

🛠️ Tech Stack

🐍 Python

📚 Scikit-learn (TF-IDF, ML models, pipelines)

⚡ Optuna (hyperparameter tuning)

🚀 Conclusion

This project demonstrates the power of AutoML + Hyperparameter Tuning for spam detection.
By automating model comparison and optimization, the final Linear SVC model achieved:

✅ 98% accuracy
✅ 0.98 weighted F1-score
✅ Robust & reliable performance for real-world spam detection
