# 📩 AutoML for Spam Detection  

## 🎯 Objective  
Use **AutoML** and **Hyperparameter Tuning** tools to automate the model selection and optimization process for **spam detection**.  

---

## 📖 Theory  
**AutoML** automates key parts of the machine learning pipeline, including:  

- 🔹 Feature extraction  
- 🔹 Model selection  
- 🔹 Model evaluation  

In this project:  

### 🔍 Models Compared  
- ⚡ Logistic Regression  
- ⚡ Multinomial Naive Bayes  
- ⚡ Linear Support Vector Classifier (SVC)  
- ⚡ Random Forest  

### 🧹 Preprocessing  
- **TF-IDF Vectorizer** was used to convert SMS messages into numerical features while:  
  - Removing stopwords  
  - Considering unigrams and bigrams  

➡️ Text-based data requires effective representation for ML models to perform well.  

---

## ⚙️ Methodology  

### 1️⃣ Model Evaluation  
- Metric: **F1-score** (balances precision & recall)  
- Reason: Best for **imbalanced datasets** like spam detection  
- ✅ **Result**: **Linear SVC** emerged as the strongest baseline  

### 2️⃣ Hyperparameter Tuning  
- Framework: **Optuna** (state-of-the-art optimization tool)  
- Parameters tuned:  
  - 📌 N-gram range  
  - 📌 Minimum document frequency (`min_df`)  
  - 📌 Regularization parameter (`C`) for SVC  

### 3️⃣ Final Model Training  
- Trained the **optimized SVC model** on the training set  
- Evaluated performance on the **test set**  

---

## 📊 Results  

| 📌 Metric       | ✅ Ham (Non-Spam) | 🚨 Spam |
|-----------------|------------------|---------|
| **Precision**   | 0.99             | 0.96    |
| **Recall**      | 0.99             | 0.91    |
| **F1-score**    | 0.99             | 0.94    |

- 🏆 **Overall Accuracy**: **98%**  
- 🏆 **Weighted F1-score**: **0.98**  

✨ The tuned model balanced:  
- ❌ False Positives (ham → spam)  
- ❌ False Negatives (spam missed)  

---

## 🛠️ Tech Stack  
- 🐍 Python  
- 📚 Scikit-learn (TF-IDF, ML models, pipelines)  
- ⚡ Optuna (hyperparameter tuning)  

---

## 🚀 Conclusion  
This project demonstrates the power of **AutoML + Hyperparameter Tuning** for spam detection.  

✅ **98% accuracy**  
✅ **0.98 weighted F1-score**  
✅ **Robust performance** for real-world applications  

---

## 📂 Project Setup  

```bash
# Clone the repository
git clone https://github.com/neeraj-jhaa/Project_AutoML.git

# Navigate into the project directory
cd Project_AutoML

# Install dependencies
pip install -r requirements.txt

# Run the training script
python train.py

