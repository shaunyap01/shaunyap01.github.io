---
title: "Evaluating Environment & Climate Truthfulness in Social Media using Deep Learning & Large Language Models (LLMs)"
date: 2024-09-01
lastmod: 2024-09-01
tags: ["`Python", "^^All Projects", "Natural Language Processing (NLP)", "Feature Engineering", "Web scrape", "Deep Learning", "Large Language Models (LLM)", "Environmental Misinformation"]
author: "Shaun Yap"
description: "" 
summary: "Awarded Best Dissertation in Cohort, this MSc project explores the detection of climate and environmental misinformation on social media using a comparative framework of traditional natural language processing techniques, deep learning, and Large Language Models (LLMs). Leveraging a web-scraped dataset from PolitiFact, the study highlights the superiority of CNNs trained on ordinal truthfulness data, with accuracy boosted from 80.1% to 84.0% through GPT-4o-driven feature augmentation. While LLMs enhanced contextual understanding and sentiment analysis, their time complexity posed practical limitations. The project contributes novel insights into model performance trade-offs, evaluation metrics tailored to ordinal classification, and the practical integration of LLMs for misinformation mitigation in climate discourse." 
cover:
    image: ""
    alt: ""
    relative: false
editPost:
showToc: true
disableAnchoredHeadings: false

---

MSc Data Science & Statistics – University of Exeter

Recipient of MSc Project Award for Best Dissertation in Cohort

---

# Research Objectives
The central aim of my Master’s dissertation is to advance the development of machine learning tools capable of identifying climate and environmental misinformation circulating on social media. In an age where accurate scientific communication is essential, this research investigates how effectively different modelling approaches can discern the truthfulness of climate-related claims.

To achieve this, the project was guided by three core objectives:

1. Evaluate the effectiveness of Large Language Models (LLMs), such as OpenAI’s GPT-4o, in detecting climate and environmental misinformation. This involved assessing their classification performance and exploring their potential to augment conventional models through advanced sentiment and readability analysis.

2. Assess the performance of other deep learning architectures, including Convolutional Neural Networks (CNNs), in classifying the truthfulness of online climate statements, particularly in comparison to LLMs and traditional NLP approaches.

3. Benchmark traditional machine learning models (e.g. random forests, gradient boosting classifiers, support vector machines) to establish foundational performance metrics and understand their limitations in handling nuanced truthfulness labels.


# Methodology

## Data Collection and Preparation
The dataset used in this research was web-scraped from a publicly available fact-checking resource [PolitiFact](https://www.politifact.com), consisting of climate- and environment-related statements classified on an ordinal truthfulness scale:
'Pants on Fire', 'False', 'Mostly False', 'Half True', 'Mostly True', and 'True'. This rich ordinal labelling allowed for a nuanced understanding of misinformation, rather than a simplistic binary approach.
To facilitate comparative analysis, two distinct datasets were engineered:
- **Ordinal Classification Dataset**: Retains the ordinal clasification and combines 'Pants on Fire' into 'False'.
- **Binary Classification Dataset**: Collapses the labels into two classes - 'True' and 'False' - following prior work in this domain.

Exploratory Data Analysis (EDA) revealed subtle patterns in language use, punctuation, and sentiment distribution across truthfulness levels. These insights informed the subsequent feature engineering and modelling strategies.

### Feature Engineering
Text-based features were derived using standard Natural Language Processing (NLP) tools, including:
- Readability: Calculated using Flesch-Kincaid reading ease scores.
- Sentiment Analysis: Utilised polarity and subjectivity scores generated via standard NLP library TextBlob.
- Custom Features: Initial trials included features such as punctuation frequency, based on observed correlations during EDA. However, this feature was excluded after statistical testing revealed limited generalisability beyond the dataset.

To further enhance feature richness, the study introduced LLM-based augmentation using OpenAI's GPT-4o. Each statement was assessed by GPT-4o for:
- Readability
- Polarity
- Subjectivity
with scores normalised and averaged over three independent passes to reduce variability in outputs. This approach aimed to provide context-aware sentiment information beyond the capabilities of rule-based NLP tools.

## Evaluation Metrics
Accurate evaluation of model performance is critical in the context of misinformation detection, where misclassifications - especially false positives - can have serious consequences. In this study, the evaluation strategy is carefully tailored to the type of task: **ordinal classification**, where labels have an inherent order, and **binary classification**, where labels are simplified into true/false categories. Different metrics were applied based on the classification type to ensure robust and meaningful assessments.

### Metrics for Ordinal Classification
In ordinal classification, statements are categorised on a five-point truthfulness scale ranging from 'False' to 'True'. Since the labels are ordered, standard accuracy metrics do not fully capture the quality of predictions. The following metrics were employed:

1. Ordinal Classification Accuracy
This metric measures the proportion of predictions that match the actual class exactly,
providing a straightforward assessment of how often the model correctly classifies
statements:

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$

Rationale: This metric offers a direct and intuitive measure of the model's overall
performance by focusing on exact matches between predicted and actual classes.

2. Ordinal Adjacent Accuracy
Ordinal Classification Adjacent Accuracy extends the concept of accuracy by considering the
proximity of the predicted class to the actual class. Specifically, we define this accuracy as the proportion of predictions that fall within one category of the true label:

$$\text{Adjacent Accuracy} = \frac{\text{Number of predictions within } \pm 1 \text{ class of actual}}{\text{Total number of predictions}}$$

Rationale: This metric recognises that not all errors are equally severe in ordinal
classification. For example, predicting "MOSTLY TRUE" when the correct label is "TRUE" is
less problematic than predicting "FALSE". By using Ordinal Classification Adjacent Accuracy,
we capture the quality of predictions in a way that respects the ordered nature of the
truthfulness ratings.

3. Mean Squared Error (MSE)
Mean Squared Error is another metric used to evaluate ordinal classification by penalising
predictions based on their distance from the actual label.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Rationale: MSE is valuable during the training and validation phases as it penalises larger
errors more heavily, helping to refine the model by minimising the distance between
predicted and true classes. This focus on reducing error magnitude makes MSE an
important metric for model optimisation. This metric will primarily be used for initial model fitting and validation.

### Metrics for Binary Classification
1. Accuracy
Accuracy is the most straightforward metric, representing the proportion of correctly
classified instances among all instances. It provides a general sense of how well the model
is performing. In the binary classification task, accuracy is calculated as:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Rationale: While accuracy provides an overall view of model performance, it may not be
sufficient in cases of imbalanced datasets or when the costs of different types of errors (i.e., false positives vs. false negatives) are not equal. However, it serves as a useful baseline metric to understand the general effectiveness of the model.

2. Precision
Precision measures the proportion of true positive predictions among all positive predictions made by the model:

$$\text{Precision} = \frac{TP}{TP + FP}$$

Rationale: Precision is particularly important in this context because it reflects the model's ability to avoid false positives, which are especially costly when dealing with misinformation. In our scenario, a high precision ensures that when the model predicts a statement as true, it is likely to be accurate, thereby minimising the spread of misinformation.

3. False Positive Rate (FPR)
The False Positive Rate is the proportion of actual negatives that are incorrectly classified as positives:
$$\text{FPR} = \frac{FP}{FP + TN}$$

Rationale: FPR is a critical metric in our application because it directly measures the rate at which false statements are incorrectly classified as true, which can lead to the propagation of misinformation. By closely monitoring the FPR, we can evaluate how well the model is mitigating the risk of false positives, which, in this context, is more damaging than false negatives. A primary objective of our modelling is to minimise the false positive rate.

4. Recall
Recall measures the proportion of actual true statements (positives) that are correctly
identified by the model. It specifically focuses on the model’s ability to capture true
statements, highlighting how well the model performs in avoiding false negatives (i.e., true
statements incorrectly classified as false).
$$\text{Recall} = \frac{TP}{TP + FN}$$

Rationale: While minimising the false positive rate is a primary objective, it is equally
important to ensure that the model is not overly conservative, thereby failing to recognise
true statements. Given the inherent trade-off between reducing type I errors (false positives) and type II errors (false negatives), tuning the model to decrease the FPR can result in a decrease in recall. A high recall ensures that the model effectively captures the true statements within the dataset, preventing valuable information from being overlooked. This is particularly vital in the context of climate change and environmental discourse, where underestimating the truthfulness of statements could lead to the dismissal of important facts and hinder public understanding.

To ensure a balanced approach, this study will require all models to achieve a minimum
recall of 50%. This threshold guarantees that at least half of all true statements are correctly identified by the model, striking a necessary balance between minimising false positives and maintaining the model's ability to accurately recognise true statements. By enforcing this minimum recall requirement, the model is held accountable not only for reducing misinformation but also for preserving the integrity of factual information, which is essential for building a trustworthy system for detecting climate-related truths and falsehoods.


## Text Vectorisation Techniques
The raw text data was transformed into numerical representations using a range of vectorisation techniques:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Bag of Words
- Word2Vec
- BERT
- Fine-tuned BERT

Among these, TF-IDF was selected as the default method for most models due to its favourable trade-off between performance and computational cost. While models incorporating BERT showed promising results, the significantly higher computational overhead limited its use in large-scale model iterations.

## Modelling Approaches
The modelling pipeline involved training and evaluating three classes of models:
1. Traditional Machine Learning Models:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- Support Vector Machines (SVM)
These models served as the baseline for performance comparison.

2. Deep learning Models:
- Covolutional Neural Networks (CNNs) trained on embedded vector representations of the text.
- Fine-tuned transformer models where appropriate.

CNNs were found to outperform traditional models, particularly on the ordinal classification dataset, which preserved more contextual truthfulness information.

3. Large Language Models (LLMs):
- GPT-4o was evaluated both as a data augmentation tool and as a standalone zero-shot classifier.

While GPT-4o augmentation marginally improved performance for traditional models (Binary accuracy +1.9%), it had a significant impact on CNNs, improving binary classification accuracy from 80.1% to 85.0%.

# Key Results
- Feature engineering showed significant increase (~20%) in binary accuracy.
- Prior work in the domain collapsed the labels in to a binary classification and this work shows that there is a significant increase in performance (~10%) when training on the ordinal classification then converting back to binary labels.
- Tested LLMs as a data augmentation tool (theorised it could provide more nuanced, context-aware scores) and standalone zero-shot classifier. GPT-4o data augmentation showed a ~2% improvement on baseline model and a ~5% improvement on CNN model.
- Trade-off: GPT-4o enhanced data augmentation quality, but massively increased time complexity raises practical concerns for large-scale or real-time applications.
