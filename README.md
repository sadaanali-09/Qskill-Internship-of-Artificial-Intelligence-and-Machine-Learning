# Qskill-Internship-of-Artificial-Intelligence-and-Machine-Learning
Projects based on Artificial Intelligence and Machine Learning implemented in Python.

1. House Price Prediction (House Price Prediction.py)
   - Description: A supervised regression pipeline that predicts median house values based on the classical California Housing dataset.
   - Key Features:
     * Loads data directly from scikit-learn.
     * Splits data into training and testing sets (80/20 ratio).
     * Applies feature scaling using StandardScaler.
     * Trains a Linear Regression model.
     * Evaluates performance using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) Score.

2. Spam Mail Detector (Spam Mail Detector.py)
   - Description: A Natural Language Processing (NLP) classification pipeline that identifies spam SMS messages using a Multinomial Naive Bayes classifier.
   - Key Features:
     * Fetches SMS spam collection data from a remote repository.
     * Converts text messages into numerical feature vectors using TF-IDF Vectorizer (with English stop-word removal).
     * Handles class distribution and stratified data splitting.
     * Trains a Naive Bayes model and outputs Accuracy, Classification Report, and Confusion Matrix.
     * Includes an interactive command-line interface (CLI) prompt allowing users to test custom messages in real-time.

--------------------------------------------------
PREREQUISITES & INSTALLATION
--------------------------------------------------

Make sure you have Python installed along with the required libraries. You can install the dependencies via pip:

pip install numpy pandas scikit-learn

--------------------------------------------------
USAGE
--------------------------------------------------

Clone the repository and run any script directly from your terminal:

# Run House Price Prediction
python "House Price Prediction.py"

# Run Spam Mail Detector
python "Spam Mail Detector.py"

--------------------------------------------------
CONTRIBUTING
--------------------------------------------------
Contributions, suggestions, and issues are welcome! Feel free to fork this repository and submit pull requests.
