# =============================
# Imports
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# =============================
# Step 1: Load Dataset
# =============================
DATA_FILE = "l_pataraia25_35791.csv"
data = pd.read_csv(DATA_FILE)

feature_cols = ["words", "links", "capital_words", "spam_word_count"]
target_col = "is_spam"

X_data = data[feature_cols]
y_data = data[target_col]


# =============================
# Step 2: Train / Test Split
# =============================
X_tr, X_te, y_tr, y_te = train_test_split(
    X_data, y_data, test_size=0.3, random_state=42
)


# =============================
# Step 3: Train Logistic Regression Model
# =============================
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_tr, y_tr)


# =============================
# Step 4: Model Coefficients
# =============================
coef_table = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": log_reg_model.coef_[0]
})

print("\nLogistic Regression Coefficients:")
print(coef_table)


# =============================
# Step 5: Model Evaluation
# =============================
y_predictions = log_reg_model.predict(X_te)

conf_matrix = confusion_matrix(y_te, y_predictions)
model_accuracy = accuracy_score(y_te, y_predictions)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nAccuracy:", model_accuracy)


# =============================
# Step 6: Confusion Matrix Heatmap
# =============================
plt.figure()
ax = sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    xticklabels=["Legitimate", "Spam"],
    yticklabels=["Legitimate", "Spam"]
)

ax.set_title("Confusion Matrix Heatmap")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# Enable coordinate display on hover
ax.format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"

plt.show()


# =============================
# Step 7: Class Distribution Plot
# =============================
class_counts = data[target_col].value_counts().sort_index()

plt.figure()
plt.bar(["Legitimate", "Spam"], class_counts.values)
plt.xlabel("Email Class")
plt.ylabel("Count")
plt.title("Class Distribution of Emails")
plt.show()


# =============================
# Step 8: Numeric Feature-Based Classification
# =============================
def predict_from_features(word_cnt, link_cnt, caps_cnt, spam_kw_cnt):
    sample_df = pd.DataFrame([{
        "words": word_cnt,
        "links": link_cnt,
        "capital_words": caps_cnt,
        "spam_word_count": spam_kw_cnt
    }])
    
    result = log_reg_model.predict(sample_df)[0]
    return "Spam" if result == 1 else "Legitimate"


# Example usage
print(predict_from_features(85, 2, 4, 1))


# =============================
# Step 9: Text-Based Email Classification
# =============================
SPAM_TERMS = {
    "free", "win", "winner", "money", "cash", "prize",
    "offer", "click", "buy", "urgent", "limited"
}

def extract_email_features(email_body):
    tokens = email_body.split()
    cleaned = [tok.strip(string.punctuation) for tok in tokens]
    
    total_words = len(cleaned)
    total_links = len(re.findall(r"(http[s]?://|www\.)", email_body.lower()))
    capitalized_words = sum(1 for w in cleaned if w.isupper() and len(w) > 1)
    spam_terms_count = sum(1 for w in cleaned if w.lower() in SPAM_TERMS)
    
    return {
        "words": total_words,
        "links": total_links,
        "capital_words": capitalized_words,
        "spam_word_count": spam_terms_count
    }


def classify_email_text(email_body):
    feature_dict = extract_email_features(email_body)
    sample_df = pd.DataFrame([feature_dict])
    
    prediction = log_reg_model.predict(sample_df)[0]
    probability = log_reg_model.predict_proba(sample_df)[0][1]
    
    print("Extracted features:", feature_dict)
    print(f"Spam probability: {probability:.3f}")
    
    return "Spam" if prediction == 1 else "Legitimate"


# =============================
# Step 10: Example Emails
# =============================
spam_txt = """
URGENT OFFER!!!
WIN BIG CASH NOW!!!
CLICK HERE TO CLAIM YOUR PRIZE!!!
Visit www.instant-win-reward.com
"""

print("Spam Email Prediction:", classify_email_text(spam_txt))


legitimate_txt = """
Hi everyone,
The project documentation has been uploaded to the shared drive.
Please review it before our next meeting.
"""


print("Legitimate Email Prediction:", classify_email_text(legitimate_txt))
