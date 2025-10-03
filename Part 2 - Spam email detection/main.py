"""
Spam Email Detection System - Logistic Regression
Machine Learning for Cybersecurity - Midterm Project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = r"C:\Users\Administrator\Desktop\NKU-GEO\Machine Learning for Cybersecurity\MIDTERM PYTHON PROJECTS\Part 2 - Spam email detection\n_asatiani2024_652847.csv"

# =============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# =============================================================================
print("=" * 80)
print("SPAM DETECTION USING LOGISTIC REGRESSION")
print("=" * 80)

# Load the dataset
print("\n[1] Loading Data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)[:5]}... (showing first 5)")

# Handle missing values
df = df.fillna(0)

# Identify target column (spam label)
target_col = None
for col in ['spam', 'label', 'class', 'target', 'is_spam']:
    if col in df.columns:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]

print(f"Target column: '{target_col}'")

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]
feature_names = list(X.columns)

print(f"Features: {len(feature_names)}")
print(f"Samples: {len(y)}")
print(f"Class distribution:\n{y.value_counts()}")

# =============================================================================
# PART 2: TRAIN-TEST SPLIT (70%-30%)
# =============================================================================
print("\n[2] Splitting Data (70% train, 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler")

# =============================================================================
# PART 3: LOGISTIC REGRESSION MODEL TRAINING
# =============================================================================
print("\n[3] Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
model.fit(X_train_scaled, y_train)
print("Model trained successfully!")

# Extract model coefficients
coefficients = model.coef_[0]
intercept = model.intercept_[0]

print("\n" + "-" * 80)
print("MODEL COEFFICIENTS")
print("-" * 80)
print(f"Intercept (β₀): {intercept:.6f}\n")

# Display top 15 most important features
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coef': np.abs(coefficients)
}).sort_values('Abs_Coef', ascending=False)

print("Top 15 Most Important Features:")
for i, row in coef_df.head(15).iterrows():
    direction = "→ SPAM" if row['Coefficient'] > 0 else "→ HAM"
    print(f"{row['Feature']:40s}: {row['Coefficient']:10.6f} {direction}")

# Training accuracy
train_pred = model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, train_pred)
print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")

# =============================================================================
# PART 4: MODEL TESTING AND EVALUATION
# =============================================================================
print("\n[4] Testing Model on Unseen Data...")
y_pred = model.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n" + "-" * 80)
print("CONFUSION MATRIX")
print("-" * 80)
print("\n              Predicted")
print("           Class 0  Class 1")
print("         +-----------------")
print(f"Actual 0 |   {cm[0][0]:4d}     {cm[0][1]:4d}")
print(f"       1 |   {cm[1][0]:4d}     {cm[1][1]:4d}")

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
accuracy = accuracy_score(y_test, y_pred)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "-" * 80)
print("PERFORMANCE METRICS")
print("-" * 80)
print(f"True Negatives (TN):  {tn:4d}  (Correct HAM predictions)")
print(f"False Positives (FP): {fp:4d}  (HAM predicted as SPAM)")
print(f"False Negatives (FN): {fn:4d}  (SPAM predicted as HAM)")
print(f"True Positives (TP):  {tp:4d}  (Correct SPAM predictions)")
print(f"\nAccuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")


# =============================================================================
# PART 4.5: DATA VISUALIZATIONS
# =============================================================================
print("\n[4.5] Generating Visualizations...")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

# -------------------------------------------------------------------------
# VISUALIZATION 1: Class Distribution with Detailed Statistics
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Count plot
class_counts = y.value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for Ham, Red for Spam
bars = ax1.bar(['Ham (0)', 'Spam (1)'], class_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(y)*100:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Number of Emails', fontsize=12, fontweight='bold')
ax1.set_xlabel('Email Class', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Email Classes in Dataset', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3)

# Pie chart
ax2.pie(class_counts.values, labels=['Ham (Legitimate)', 'Spam (Unwanted)'],
        autopct='%1.1f%%', colors=colors, startangle=90,
        explode=(0.05, 0.05), shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Proportional Class Distribution', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()

# Save with explicit path in the same folder as the CSV
save_path_1 = r"C:\Users\Administrator\Desktop\NKU-GEO\Machine Learning for Cybersecurity\MIDTERM PYTHON PROJECTS\Part 2 - Spam email detection\class_distribution.png"
plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
print(f"✓ Visualization 1 saved: {save_path_1}")
plt.close()

# -------------------------------------------------------------------------
# VISUALIZATION 2: Top Features Importance (Coefficient Analysis)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Get top 15 features by absolute coefficient value
top_features = coef_df.head(15).copy()
top_features = top_features.sort_values('Coefficient')  # Sort for better visualization

# Create color map: positive = spam indicators (red), negative = ham indicators (green)
colors_features = ['#e74c3c' if x > 0 else '#2ecc71' for x in top_features['Coefficient']]

# Horizontal bar chart
bars = ax.barh(range(len(top_features)), top_features['Coefficient'], color=colors_features, alpha=0.7, edgecolor='black', linewidth=1.2)

# Customize
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('Coefficient Value (Impact on Spam Prediction)', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Most Important Features for Spam Detection\n(Red = Spam Indicators, Green = Ham Indicators)',
             fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_features['Coefficient'])):
    label_x = val + (0.02 if val > 0 else -0.02)
    ha = 'left' if val > 0 else 'right'
    ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
            ha=ha, va='center', fontsize=9, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='Spam Indicators (→ Spam)'),
                   Patch(facecolor='#2ecc71', alpha=0.7, edgecolor='black', label='Ham Indicators (→ Legitimate)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()

# Save with explicit path in the same folder as the CSV
save_path_2 = r"C:\Users\Administrator\Desktop\NKU-GEO\Machine Learning for Cybersecurity\MIDTERM PYTHON PROJECTS\Part 2 - Spam email detection\feature_importance.png"
plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
print(f"✓ Visualization 2 saved: {save_path_2}")
plt.close()

print("\n" + "=" * 60)
print("VISUALIZATION SUMMARY")
print("=" * 60)
print(f"Graph 1: Class Distribution")
print(f"  - Shows balance between Ham and Spam emails")
print(f"  - Total samples: {len(y)}")
print(f"  - Ham: {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
print(f"  - Spam: {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")
print(f"\nGraph 2: Feature Importance")
print(f"  - Displays top 15 features affecting spam prediction")
print(f"  - Positive coefficients → Increase spam probability")
print(f"  - Negative coefficients → Indicate legitimate email")
print(f"  - Strongest spam indicator: {coef_df.iloc[0]['Feature']}")
print("=" * 60)


# =============================================================================
# PART 5: EMAIL SPAM CHECKER FUNCTION
# =============================================================================
def check_email_spam(email_text, model, scaler, feature_names):
    """
    Extract features from email text and predict if spam
    Features: words, links, capital_words, spam_word_count
    """
    import re

    # Initialize features dictionary
    features = {}

    email_lower = email_text.lower()
    words_list = email_text.split()

    # 1. Count total words
    features['words'] = len(words_list)

    # 2. Count links (URLs)
    url_pattern = r'http[s]?://|www\.'
    features['links'] = len(re.findall(url_pattern, email_lower))

    # 3. Count capital words (words that are ALL CAPS or start with capital)
    capital_words = 0
    for word in words_list:
        # Check if word has capital letters
        if any(c.isupper() for c in word):
            capital_words += 1
    features['capital_words'] = capital_words

    # 4. Count spam keywords
    spam_keywords = [
        'free', 'winner', 'win', 'won', 'cash', 'prize', 'urgent', 'click',
        'buy', 'offer', 'discount', 'guarantee', 'credit', 'money', 'earn',
        'limited', 'now', 'act', 'order', 'claim', 'congratulations',
        'bonus', 'gift', 'deal', 'save', 'cheap', 'best', 'amazing'
    ]

    spam_word_count = 0
    for keyword in spam_keywords:
        spam_word_count += email_lower.count(keyword)
    features['spam_word_count'] = spam_word_count

    # Print extracted features for debugging
    print(f"\nExtracted features:")
    print(f"  Words: {features['words']}")
    print(f"  Links: {features['links']}")
    print(f"  Capital words: {features['capital_words']}")
    print(f"  Spam word count: {features['spam_word_count']}")

    # Convert to DataFrame in correct order
    feature_vector = pd.DataFrame([features])[feature_names]

    # Scale features
    feature_scaled = scaler.transform(feature_vector)

    # Predict
    prediction = model.predict(feature_scaled)[0]
    probability = model.predict_proba(feature_scaled)[0]

    return prediction, probability


# =============================================================================
# PART 6: INTERACTIVE TESTING
# =============================================================================
print("\n" + "=" * 80)
print("INTERACTIVE EMAIL CHECKER")
print("=" * 80)

# Sample spam email
sample_spam = """
URGENT WINNER ALERT!!! You have WON $1,000,000 CASH PRIZE absolutely FREE!!!
Click here NOW to claim your AMAZING offer! LIMITED time only!!!
Send your credit card details to winner@free-money.com to claim your prize!
Act fast! This is a GUARANTEED winner offer! FREE money! Best deal ever!
Order now! Save big! Earn cash! Buy now! Discount available! 
Click click click! Amazing prize! Free free free! Winner winner!
"""

# Sample legitimate email
sample_ham = """
Hi John,
I hope this email finds you well. I wanted to follow up on our meeting
from last Tuesday regarding the project timeline.
Best regards,
Sarah
"""

while True:
    print("\nOptions:")
    print("1. Test sample SPAM email")
    print("2. Test sample HAM email")
    print("3. Enter custom email text")
    print("4. Exit")

    choice = input("\nChoice (1-4): ").strip()

    if choice == '1':
        email = sample_spam
        print("\n[Testing SPAM sample]")
    elif choice == '2':
        email = sample_ham
        print("\n[Testing HAM sample]")
    elif choice == '3':
        print("\nEnter email text (press Enter twice when finished):")
        lines = []
        empty_count = 0
        while True:
            line = input()
            if line.strip() == '':
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                lines.append(line)
        email = '\n'.join(lines)
        if not email.strip():
            print("No text entered!")
            continue
    elif choice == '4':
        print("\nThank you for using Spam Detection System!")
        break
    else:
        print("Invalid choice!")
        continue

    if choice in ['1', '2', '3']:
        print(f"\nEmail preview:\n{'-' * 60}\n{email[:150]}...")
        print('-' * 60)

        pred, prob = check_email_spam(email, model, scaler, feature_names)

        print("\n" + "=" * 60)
        if pred == 1:
            print("⚠️  SPAM DETECTED!")
            print(f"Spam probability:  {prob[1] * 100:.2f}%")
            print(f"Ham probability:   {prob[0] * 100:.2f}%")
        else:
            print("✓ LEGITIMATE EMAIL (HAM)")
            print(f"Ham probability:   {prob[0] * 100:.2f}%")
            print(f"Spam probability:  {prob[1] * 100:.2f}%")
        print("=" * 60)

print("\n" + "=" * 80)
print("PROGRAM COMPLETED")
print("=" * 80)