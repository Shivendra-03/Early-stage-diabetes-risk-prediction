# Early-Stage Diabetes Risk Prediction

A small end-to-end data science project that asks two things: which symptoms actually go hand in hand with diabetes, and can we predict someone's risk from them?

## Overview

I used the [Early Stage Diabetes Risk Prediction dataset](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) from the UCI Machine Learning Repository. It has 520 patient records with 16 features (age, gender and a bunch of yes/no symptoms like polyuria, polydipsia and sudden weight loss) plus a positive/negative diabetes label.

The project has two parts:

1. **EDA and statistical testing** – Chi-square tests for the categorical features and a Mann-Whitney U test for age, to see which ones are significantly linked to diabetes.
2. **Predictive modelling** – Six classifiers, tuned with grid search and compared on accuracy, precision, recall, F1 and MCC.

## Folder Structure

```
.
├── data/
│   └── diabetes_data_upload.csv
├── notebooks/
│   ├── EDA.ipynb   
│   ├── Predictive_Classification_Analytics.ipynb                       
│   └── conclution.ipynb                                  
├── requirements.txt
└── README.md
```

> The notebooks read the data from `data\diabetes_data_upload.csv`, so download the CSV from the UCI link above. 

## Tech Stack

- **Python 3**
- **pandas, NumPy** – data handling
- **Matplotlib, Seaborn** – visualisation
- **SciPy** – Chi-square and Mann-Whitney U tests
- **scikit-learn** – preprocessing (one-hot encoding, scaling), models, grid search, metrics
- **Jupyter Notebook**

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

## Questions Explored

- Is gender associated with the likelihood of diabetes?
- Do polyuria and polydipsia (excess urination and thirst) show a significant association?
- Does sudden weight loss relate to diabetes status?
- Are itching and delayed healing actually linked to diabetes?
- Is there a significant age difference between diabetic and non-diabetic people?
- Which classification model predicts diabetes risk best?

## Conclusion

**Statistical findings:** Gender, polyuria, polydipsia, sudden weight loss, weakness, polyphagia, genital thrush, visual blurring, irritability, partial paresis, muscle stiffness and alopecia are all significantly associated with diabetes. Itching and delayed healing showed no conclusive association. Age also differed significantly between the diabetic and non-diabetic groups.

**Model results:**

| Model | Takeaway |
|---|---|
| K-Nearest Neighbors | Best overall: 96.15% accuracy, 96.19% precision |
| Decision Tree / Random Forest | Close behind at about 94.23% accuracy and 94.43% precision |
| SVM | Solid across accuracy, precision and MCC |
| Naive Bayes | 91.35% accuracy, but lower precision and MCC |
| Logistic Regression | 87.50% accuracy, weakest of the group |

KNN came out on top, though with only 520 records it's worth validating on more data before reading too much into the gap between models.
