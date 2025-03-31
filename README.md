# Classifying transcribed speech from interviews or media reports by key topics

# Course: NLP (Semester 6) - Pillai College of Engineering
# Project Overview
This project focused on intent classification in chatbot systems using Natural Language Processing (NLP) and deep learning techniques. Various preprocessing methods such as tokenization, stopword removal, and word embeddings were applied to improve text representation. The study explored different machine learning models (Logistic Regression, Random Forest, SVM, Naïve Bayes, KNN) and deep learning architectures (CNN, LSTM, CNN-BiLSTM), comparing their performance across multiple feature extraction techniques like BoW, TF-IDF, and FastText. Transformer-based models BERT and RoBERTa demonstrated superior accuracy and classification performance, significantly outperforming traditional and deep learning methods. The experimental results highlight the effectiveness of advanced NLP models in enhancing chatbot understanding, user interaction, and response accuracy.

You can learn more about the college by visiting the official website of Pillai College of Engineering.

# Acknowledgements:
We would like to express our sincere gratitude to the following individuals:
Theory Faculty:
Dhiraj Amin
Lab Faculty:
Dhiraj Amin
Neha Ashok
Their guidance and support have been invaluable throughout this project.

# Project Title: Classifying transcribed speech from interviews or media reports by key topics. This process aids journalists, content creators, and analysts by summarizing and categorizing important themes, improving content searchability and reporting efficiency.

# Project Abstract:
This project builds a text classification model to sort transcribed speech into topics like Cybersecurity, Science, Stock Market, Climate, Politics, and Software Engineering. Using advanced Natural Language Processing (NLP) techniques and machine learning, the model processes raw text, extracts key features, and assigns it to the appropriate category. It aims to assist journalists, analysts, and researchers by enabling them to quickly identify relevant themes in large datasets, saving time and improving workflow efficiency. The model leverages CountVectorizer for transforming text into numerical representations, capturing the frequency of words, and facilitating effective analysis. Naive Bayes, a probabilistic classifier, is utilized for its speed and accuracy in categorizing text based on word distributions. The model is trained on labeled transcribed speech datasets, ensuring its ability to classify unseen data with high precision. This approach not only simplifies content organization but also has wide applications in automated content tagging, real-time analytics, and enhancing research productivity. Furthermore, the model’s scalability allows it to be easily extended for additional topics or languages. By automating the classification of transcribed speech, the model helps improve the overall speed of content filtering and analysis, making it highly valuable for organizations dealing with large volumes of data. Additionally, it can support multiple industries, such as media, finance, and climate research, offering a versatile tool for any domain that requires efficient text categorization.

# Algorithms Used:
Machine Learning Algorithms:

Logistic Regression
Support Vector Machine (SVM)
Random Forest Classifier

Deep Learning Algorithms:

Convolutional Neural Networks (CNN)
Long Short-Term Memory (LSTM)
CNN-BiLSTM


Language Models:

ROBERTA
BERT (Bidirectional Encoder Representations from Transformers)

# Comparative Analysis:

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

# Conclusion:
Classifying transcribed speech using NLP enhances journalism by automating topic identification and improving content organization. While ML models offer efficiency, deep learning and transformer-based models like BERT and GPT provide superior accuracy and contextual understanding.
Through comparative analysis, language models emerge as the best choice for media applications due to their adaptability. Future improvements can focus on real-time processing, multimodal integration, and domain-specific fine-tuning to further enhance classification accuracy and efficiency, ultimately streamlining media workflows and decision-making.


RoBERTa: Performed significantly lower, with an accuracy of 40%.
![image](https://github.com/user-attachments/assets/c038de89-5433-4a8f-9ac8-05bafa0108e0)
