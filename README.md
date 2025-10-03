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

4. 
The application has the built-in capability to evaluate new email texts for spam. The script contains a dedicated function named check_email_spam which is designed specifically for this purpose. This function correctly parses raw email text provided by a user, extracts the exact same features the model was trained on—specifically words, links, capital_words, and spam_word_count—and then uses the already trained logistic regression model and scaler to make a real-time prediction. The interactive menu in the final section of the code allows a user to easily input any custom email, which is then processed by this function to deliver a final classification of "SPAM" or "LEGITIMATE".

5. 

Next to test the applicaiton I will Compose an email text (manually) that should be (I hope) classified by my model as spam.

Email: 
Subject: URGENT: Congratulations Nika Asatiani - You are our SANGU Winner!

Dear Nika Asatiani,

AMAZING NEWS! You have won a GUARANTEED cash prize of $5,000! This is a limited time offer exclusively for SANGU students. Your student ID was selected in our monthly draw. This is a FREE gift, with no purchase necessary.

To claim your prize, you must act NOW! This deal is URGENT. Click this link immediately to verify your details: www.sangu-rewards-claim-now.com. This offer will expire in 24 hours. Don't miss this chance to earn easy money.

This is the best prize we have ever given. Order your prize confirmation by providing your credit details on our secure portal. Click here: http://secure-winner-portal.net/claim/ID=123. It is 100% FREE! We guarantee you will receive your cash bonus. Congratulations once again! Buy yourself something nice. This is a fantastic deal. Save yourself from financial worries. Win BIG today!

Act NOW! This is a LIMITED offer. FREE money is waiting for you!

Click, click, click!

Sincerely,
The Prize Redemption Team


EMAIL CREATION EXPLANATION: 

The email was written to trigger the specific features logistic regression model is sensitive to, as defined in the check_email_spam function:

High spam_word_count: The text is saturated with keywords from script's spam_keywords list. I included words like winner, prize, cash, free, urgent, click, amazing, offer, limited, guarantee, money, earn, bonus, deal, best, and buy multiple times to dramatically increase this feature's value.

Inflated capital_words Count: I used excessive and unnecessary capitalization (e.g., "URGENT", "AMAZING NEWS", "FREE", "GUARANTEED", "NOW") to increase the capital_words count, which is a common trait in spam emails.

Presence of links: Two suspicious-looking links (www.sangu-rewards-claim-now.com and http://secure-winner-portal.net/...) were included to increment the links feature count.

Sense of Urgency and "Too Good to Be True" Offer: The language creates a false sense of urgency ("Act NOW!", "limited time offer") and presents an unrealistic reward ("$5,000 cash prize"), which are classic spam tactics.



Result: 

⚠️  SPAM DETECTED!
Spam probability:  100.00%
Ham probability:   0.00% 

I am happy it worked :) 

6.

Of coourse, I will also need to compose an email text (manually) that should be (I hope) classified by my model as HAM (NOT SPAM)


Dear Nika Asatiani,

I hope your week is going well.

I have finished reviewing the initial proposal you submitted for your final project in the Machine Learning course. Your topic is quite interesting and relevant. The outline is well-structured, and your initial list of sources is solid.

I have a few suggestions regarding the methodology section that I think could strengthen your research. Could we schedule a brief 15-minute meeting next week to discuss them? Please let me know what day and time works best for you.

Keep up the excellent work.

Email Creation Methodology: 

The core principle behind the creation of this email was strategic feature minimization. The goal was to craft a message that was professional, contextually relevant, and, most importantly, generated the lowest possible values for the specific features the model is trained to recognize as spam indicators. The process involved carefully considering each of the four features and intentionally avoiding common spam triggers.

First and foremost, the vocabulary was deliberately chosen to achieve a near-zero spam_word_count. The email's content was restricted to professional and academic language directly related to the subject of a university project. Words like "reviewing," "proposal," "methodology," and "research" were used to establish a legitimate context. I consciously avoided every word on the model's spam keyword list, such as "offer," "deal," "guarantee," or "amazing." The one flagged word, "best," was used in a common phrase ("works best for you"), demonstrating how even carefully crafted emails can trigger a simple keyword counter.

I also did not include links at all as currently the model I believe would flag all links as it does not yet have link recognition capabilities built in.


In this case The application outputed the following result: 

✓ LEGITIMATE EMAIL (HAM)
Ham probability:   77.29%
Spam probability:  22.71%

I will provide few thoughts why HAM probability is less then 100%. 

The most significant factor was the Spam word count of 4. The model's keyword list contains common words that can appear in legitimate contexts. For example, the email contains the word "best" ("...works best for you"), which is on the spam list. The other three flagged words are likely similar innocuous words that the model cannot distinguish from their use in actual spam. Because the spam_word_count feature has a strong positive coefficient, even a small count significantly increases the calculated probability of spam.

The model also registered 11 Capital words and 89 Words. According to the model's coefficients from its training, an increase in any feature pushes the prediction towards spam. The 11 capital words, which came from normal sentence beginnings and proper nouns ("Nika Asatiani"), still contributed to a higher spam score. The total word count also added a small amount to the spam probability.

In short, the model correctly classified the email as legitimate but assigned a 22.71% spam probability as a measure of "uncertainty." This uncertainty was caused by the presence of a few flagged keywords and the normal use of capitalization that the model has been trained to view with suspicion.


7. 








