# nlp
# Machine Learning (ML) Models
The following traditional ML models were evaluated for text classification:

Random Forest: Achieved high accuracy (95%) using Bag-of-Words (BoW).

Support Vector Machine (SVM): Performed moderately well, but lower compared to other models.

Logistic Regression: Provided good performance with BoW features.

K-Nearest Neighbors (KNN): Similar performance to Random Forest with BoW.

Naïve Bayes: Showed lower performance compared to other models.
![image](https://github.com/user-attachments/assets/d516575d-4b27-4d97-be78-3542004ecd4f)


# Deep Learning (DL) Models
We tested deep learning architectures for text classification:

CNN (Convolutional Neural Network): Used Conv1D layers, MaxPooling, and the Adam optimizer with BoW features. Achieved an accuracy of 88%.

LSTM (Long Short-Term Memory): Leveraged TF-IDF features, dropout layers, and the Adam optimizer, reaching 0.90 accuracy.

CNN-BiLSTM: Combined CNN and BiLSTM layers, using BoW and NLP features, yielding the highest accuracy (91%).
![image](https://github.com/user-attachments/assets/87b2a174-b361-41ea-8cef-9988afe980de)


# Language Models
Pretrained transformer models were also evaluated on transcribed speech data:

BERT: Achieved strong performance with 90% accuracy, 91% precision, and 89% F1-score.

RoBERTa: Performed significantly lower, with an accuracy of 40%.
![image](https://github.com/user-attachments/assets/c038de89-5433-4a8f-9ac8-05bafa0108e0)
