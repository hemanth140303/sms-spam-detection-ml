# SMS Spam Detection Using Machine Learning

A machine learning project that classifies SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP), TF-IDF vectorization, and Multinomial Naive Bayes.

## Project Overview

Spam messages are a common problem in digital communication. This project uses machine learning to automatically identify unwanted messages and distinguish them from legitimate messages.

The model learns patterns from labeled SMS messages and predicts whether a new message is spam or legitimate.

> **Note:** This project uses an SMS spam dataset, not a dedicated email dataset.

## Dataset

- **Dataset:** UCI SMS Spam Collection
- **Original dataset size:** 5,574 messages
- **Cleaned dataset:** 5,160 unique messages
- **Legitimate messages (Ham):** 4,518
- **Spam messages:** 642

The dataset contains text messages labeled as either `ham` or `spam`.

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Matplotlib / Seaborn (for visualizations)

## Machine Learning Workflow

1. Dataset collection and loading
2. Data cleaning and duplicate removal
3. Exploratory data analysis
4. Label encoding
5. Train-test splitting
6. Text feature extraction using TF-IDF
7. Model training using Multinomial Naive Bayes
8. Model evaluation
9. Custom message predictions
10. Saving the trained model using Joblib

## Model Performance

The trained model achieved the following results on the test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 95.54% |
| Spam Precision | 100% |
| Spam Recall | 64% |

### Understanding the Results

- **Accuracy (95.54%):** Overall percentage of correctly classified test messages.
- **Precision (100%):** No legitimate messages in the test set were incorrectly classified as spam.
- **Recall (64%):** The model detected 82 out of 128 spam messages and missed 46.

Although the model achieved high precision, improving spam recall is a potential area for future development.

## Sample Predictions

The custom prediction function was tested with different messages:

| Sample Message | Prediction |
|---|---|
| Free prize message | Spam |
| Meeting tomorrow | Legitimate |
| Package delivery message | Legitimate |

The free-prize example received an estimated spam probability of 84.1%.

## Project Structure

```text
sms-spam-detection-ml/
│
├── Email_Spam_Detection_Google_Colab.ipynb
├── spam_detection_model.joblib
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/hemanth140303/sms-spam-detection-ml.git
```

### 2. Open the project folder

```bash
cd sms-spam-detection-ml
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Load the trained model

```python
import joblib

model = joblib.load("spam_detection_model.joblib")

message = ["Congratulations! You have won a free prize"]

prediction = model.predict(message)

print("Spam" if prediction[0] == 1 else "Legitimate")
```

The saved pipeline is intended to perform the same text preprocessing and prediction steps used during training.

## Future Improvements

- Improve spam recall and F1-score.
- Experiment with Logistic Regression and other classifiers.
- Test different TF-IDF configurations.
- Evaluate the model using additional datasets.
- Extend the project to genuine email datasets.
- Develop a simple web application for real-time message classification.

## Author

**Hemanth Reddy**

- GitHub: [hemanth140303](https://github.com/hemanth140303)

---

If you find this project useful, consider giving the repository a ⭐.
