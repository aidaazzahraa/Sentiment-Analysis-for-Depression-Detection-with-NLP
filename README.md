# Sentiment-Analysis-for-Depression-Detection-with-NLP
This project applies Natural Language Processing (NLP) techniques to detect depression from textual data through sentiment analysis. The process includes text preprocessing, feature extraction, model training, and evaluation to classify depressive and non-depressive texts.

## Table of Contents
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Techniques and Tools Used](#techniques-and-tools-used)
- [Outcome](#outcome)
- [Technologies Used](#technologies-used)

## Dataset

The dataset used in this project is sourced from Kaggle: [Short-Text Dataset for Depression Classification](https://www.kaggle.com/datasets/lekhnath/short-text-dataset-for-depression-classification). It consists of short text samples labeled to indicate whether they exhibit depressive tendencies. This dataset is utilized for training and evaluating classification models to detect depression through sentiment analysis.

### Dataset Features:
- **Text**: The main body of text representing user input.
- **Label**: A binary classification label where 0 indicates Non-Depressive and 1 indicates Depressive.

## Exploratory Data Analysis (EDA)

To gain a better understanding of the dataset, an exploratory data analysis (EDA) was conducted, including visualizing the distribution of depressive and non-depressive texts. The dataset consists of text samples labeled as Depressive (1) or Non-Depressive (0), and their distribution provides insights into the balance of the dataset.

### Sentiment Distribution:

The following bar chart illustrates the count of reviews for each sentiment:

- **Non-Depressive (0)**: Represents text samples that do not exhibit depressive tendencies
- **Depressive (1)**: Represents text samples that indicate depressive tendencies.

This visualization helps in identifying whether the dataset is balanced or imbalanced, which could impact model performance and necessitate data balancing techniques if needed.

![label](https://github.com/user-attachments/assets/68d402da-6aa5-4262-b620-2515bf050032)

## Techniques and Tools Used

### Text Preprocessing:
- **Case Folding**: Converting all text to lowercase to maintain uniformity and avoid case-sensitive discrepancies.
- **Word Normalization**: Standardizing non-standard words (e.g., slang or misspellings) into their standard form.
- **Stopword Removal**: Eliminating common words (e.g., "the", "and", "is") that do not contribute to the meaning of the text.
- **Stemming**: Reducing words to their root form (e.g., "running" becomes "run").

### Feature Extraction:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Used to convert the text data into numerical features based on the importance of each word in the corpus.
- **Feature Selection (SelectKBest)**: Applied Chi-square tests to select the most relevant features from the TF-IDF results, reducing the dimensionality and improving model performance.

### Machine Learning Models:
- **Naive Bayes**: A probabilistic classifier that assumes the features are independent. It provides a simple yet effective model for text classification.
- **Support Vector Machine (SVM)**: A powerful classification algorithm for high-dimensional data.
- **Random Forest**: An ensemble learning method that combines multiple decision trees for better prediction accuracy.

### Model Evaluation:
- **Accuracy**: The percentage of correct predictions out of the total predictions.
- **Confusion Matrix**: A detailed breakdown of predictions, showing true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **Cross-Validation**: Used to assess the performance of the models across different data splits to ensure robustness.

### Deployment:
- The final **Naive Bayes** model was deployed for automated sentiment prediction on new depression-related text, with real-time preprocessing to handle user inputs efficiently.
  
## Outcome:
This project successfully built a model that can classify text as indicative of depression or not, achieving an accuracy of 99%. The deployment enables automatic detection of depression-related text, providing a valuable tool for mental health monitoring and analysis.

![Model Accuracy Comparison](https://github.com/user-attachments/assets/068d4686-9d64-4dcf-8b9a-9937309ba1b3)

![Screenshot (1498)](https://github.com/user-attachments/assets/9e94a25e-dde1-4260-8aa7-5dfa737dbe07)

## Technologies Used:
- **Python** (libraries such as scikit-learn, pandas, nltk)
- **Google Colab**
- **Machine Learning Algorithms** (Naive Bayes, Support Vector Machine (SVM), Random Forest)
- **TF-IDF** for feature extraction
- **Model evaluation metrics** (accuracy, confusion matrix, classification report)
