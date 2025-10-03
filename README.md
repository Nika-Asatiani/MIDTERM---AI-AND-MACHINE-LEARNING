# MIDTERM---AI-AND-MACHINE-LEARNING

Part 1: Linear Regression and Correlation Analysis

This is part 1 of the repository. The goal is to calculate and understand correlation and linear regression. I wrote a Python script that shows both manual and automatic calculations. The work is simple to repeat and the answers are clear in the code and in the console output.

The x and y coordinates were taken manually from this website:
max.ge/aiml_midterm/652847_html

The script does the following steps:

Calculate the mean of x and y.

Calculate covariance and standard deviation.

Calculate Pearson correlation manually.

Interpret the correlation as strong, moderate, or weak.

Use scipy linregress to calculate slope, intercept, correlation, p-value, and error.

Compare manual and automatic results.

Plot the scatter data with the regression line.

Save the figure as regression_plot.png in the same folder.

To reproduce the work you need Python 3 with matplotlib, numpy, and scipy installed. Run the file regression_analysis.py in PyCharm or from the command line. The console will print the calculations and the figure will be saved as regression_plot.png.

Part 2: SPAM Email Detection

1.

Data File (named - n_asatiani2024_652847.csv) is provided in this same github repository in the following PATH:
MIDTERM---AI-AND-MACHINE-LEARNING
/Part 2 - Spam email detection/

2. 
- I created this applicaiton (source code is provided) and it is provided in this repository as main.py app inside Part 2 - Spam email detection Folder.


Data Loading and Initial Inspection
The process begins by using the powerful pandas library, a standard tool in data science for handling tabular data. The first active line of code in this section is responsible for loading the dataset from its specific location on the computer into the program.


df = pd.read_csv(DATA_PATH)
This command reads the n_asatiani2024_652847.csv file and converts it into a pandas DataFrame, which is essentially a structured table, assigned to the variable df. Immediately after loading, the script performs a quick check to understand the scale of the data.


print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)[:5]}... (showing first 5)")

The df.shape attribute provides a snapshot of the dataset's dimensions (number of rows, number of columns). Printing this is the first step in verifying that the data has been loaded correctly and gives an immediate sense of its size. Showing the first five column names helps confirm the features that are present.


Data Cleaning and Target Column Identification

Raw datasets are often imperfect and may contain missing values (represented as NaN). Machine learning models require complete numerical data to function, so these gaps must be filled. THE script handles this with a straightforward and effective cleaning step.


df = df.fillna(0)
This line finds every cell in the DataFrame that is empty and replaces it with the number 0. This ensures that the model won't encounter errors and has a complete dataset to work with. Following the cleaning, the script intelligently identifies the target column—the one that contains the labels indicating whether an email is spam (1) or not (0).

target_col = None
for col in ['spam', 'label', 'class', 'target', 'is_spam']:
    if col in df.columns:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]
    
This loop makes the code more robust and reusable. Instead of assuming the target column is always named 'spam', it checks for a list of common names. If it finds one, it assigns that name to the target_col variable. If no common name is found, it defaults to assuming the very last column in the table is the target.

Separating Features and Target for Training
Once the data is clean and the target column is known, the final preprocessing step is to separate the data into two distinct parts: features (X) and the target (y). This is a fundamental requirement for supervised machine learning.

The features (X) are the input variables—all the characteristics of the emails (like word counts, frequency of certain symbols, etc.) that the model will use to make a decision.


X = df.drop(columns=[target_col])
This code creates a new DataFrame X that includes every column from the original data except for the target column.

The target (y) is the "answer key" that the model will learn from. It contains the correct labels that correspond to each row in X.


y = df[target_col]
This line creates the y variable, which holds only the values from the spam label column. Finally, the script prints a summary of the prepared data, including a count of each class in the target variable.


print(f"Class distribution:\n{y.value_counts()}")
This command shows exactly how many emails are labeled as spam and how many are not. It's an important final check to see if the dataset is balanced or if one class significantly outnumbers the other.

3. 

After training, the model's performance is evaluated on the unseen test data. The results are summarized in a Confusion Matrix, which provides a detailed breakdown of correct and incorrect predictions.

Confusion Matrix Layout
The matrix shows how the model's predictions align with the actual labels. "Class 0" typically represents legitimate emails (Ham), and "Class 1" represents spam emails.

  Predicted
           Class 0  Class 1
         +-----------------
Actual 0 |    TN       FP
       1 |    FN       TP

True Negatives (TN): The number of legitimate emails that were correctly identified as legitimate.

False Positives (FP): The number of legitimate emails that were incorrectly identified as spam. (This is a critical error, as it means important emails might be missed).

False Negatives (FN): The number of spam emails that were incorrectly identified as legitimate. (This means spam gets through to the inbox).

True Positives (TP): The number of spam emails that were correctly identified as spam.

Key Performance Metric: Accuracy
From these values, we can calculate the overall Accuracy, which measures the proportion of total predictions that were correct.

Accuracy: The primary metric for overall performance, calculated as (TN + TP) / (TN + FP + FN + TP). It tells us, out of all emails, what percentage the model classified correctly. For example, an accuracy of 92.50% means the model was correct on 92.5% of the test emails.

Code Explanation

The evaluation process is handled in PART 4 of the script. It uses the powerful scikit-learn library to make predictions and calculate the metrics efficiently.

First, the script uses the trained model to make predictions on the scaled test features, X_test_scaled.

y_pred = model.predict(X_test_scaled)
The model.predict() function returns an array, y_pred, containing the model's classification (0 or 1) for each email in the test set.

Next, this array of predictions is compared against the true labels (y_test) to generate the confusion matrix.


cm = confusion_matrix(y_test, y_pred)
The confusion_matrix() function from scikit-learn does all the hard work, counting the TNs, FPs, FNs, and TPs and organizing them into a 2x2 matrix named cm. To make these values easier to work with, they are extracted from the matrix into individual variables.

tn, fp, fn, tp = cm.ravel()
The .ravel() method flattens the matrix into a simple list [tn, fp, fn, tp], which allows for easy assignment to the four variables. Finally, the overall accuracy is calculated using a dedicated function.

accuracy = accuracy_score(y_test, y_pred)
This accuracy_score() function compares the predicted labels with the true labels and returns the percentage of correct predictions, providing a clear, high-level summary of the model's performance.



