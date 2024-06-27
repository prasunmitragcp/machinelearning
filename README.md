# Machine Learning(Codes and Datasets)

Welcome to the Machine Learning Repository! This repository contains various machine learning models implemented in Jupyter notebooks (.ipynb files). You can run these notebooks using Google Colab.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Folder Structure](##folder-structure)
4. [Running Notebooks on Google Colab](#running-notebooks-on-google-colab)
5. [Details of Each Module](#details-of-each-module)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

This repository contains implementations of various machine learning models. Each model is contained within its own Jupyter notebook for easy experimentation and modification.

## Prerequisites

- A Google account for accessing Google Colab
- Basic knowledge of Python and machine learning concepts

## Folder Structure
```
├── Machine Learning(Codes and Datasets)
│   ├── Association Rule Learning
│   │   ├── Apriori
│   │   │   ├── Market_Basket_Optimisation.csv
│   │   │   ├── README.md
│   │   │   └── apriori.ipynb
│   │   └── Eclat
│   │       ├── Market_Basket_Optimisation.csv
│   │       ├── README.md
│   │       └── eclat.ipynb
│   ├── Classification
│   │   ├── Decision Tree Classification
│   │   │   └── Python
│   │   │       ├── Color Blind Friendly Images
│   │   │       │   ├── decision_tree_classification_test_set.png
│   │   │       │   └── decision_tree_classification_training_set.png
│   │   │       ├── README.md
│   │   │       ├── Social_Network_Ads.csv
│   │   │       └── decision_tree_classification.ipynb
│   │   ├── K-Nearest Neighbors (K-NN)
│   │   │   └── Python
│   │   │       ├── Color Blind Friendly Images
│   │   │       │   ├── knn_test_set.png
│   │   │       │   └── knn_training_set.png
│   │   │       ├── README.md
│   │   │       ├── Social_Network_Ads.csv
│   │   │       └── k_nearest_neighbors.ipynb
│   │   ├── Kernel SVM
│   │   │   └── Python
│   │   │       ├── Color Blind Friendly Images
│   │   │       │   ├── kernel_svm_test_set.png
│   │   │       │   └── kernel_svm_training_set.png
│   │   │       ├── README.md
│   │   │       ├── Social_Network_Ads.csv
│   │   │       └── kernel_svm.ipynb
│   │   ├── Logistic Regression
│   │   │   ├── Color Blind Friendly Images
│   │   │   │   ├── logistic_regression_test_set.png
│   │   │   │   └── logistic_regression_training_set.png
│   │   │   ├── README.md
│   │   │   ├── Social_Network_Ads.csv
│   │   │   └── logistic_regression.ipynb
│   │   ├── Naive Bayes
│   │   │   └── Python
│   │   │       ├── Color Blind Friendly Images
│   │   │       │   ├── naive_bayes_test_set.png
│   │   │       │   └── naive_bayes_training_set.png
│   │   │       ├── README.md
│   │   │       ├── Social_Network_Ads.csv
│   │   │       └── naive_bayes.ipynb
│   │   ├── Random Forest Classification
│   │   │   └── Python
│   │   │       ├── Color Blind Friendly Images
│   │   │       │   ├── random_forest_classification_test_set.png
│   │   │       │   └── random_forest_classification_training_set.png
│   │   │       ├── README.md
│   │   │       ├── Social_Network_Ads.csv
│   │   │       └── random_forest_classification.ipynb
│   │   └── Support Vector Machine (SVM)
│   │       └── Python
│   │           ├── Color Blind Friendly Images
│   │           │   ├── svm_test_set.png
│   │           │   └── svm_training_set.png
│   │           ├── README.md
│   │           ├── Social_Network_Ads.csv
│   │           └── support_vector_machine.ipynb
│   ├── Clustering
│   │   ├── Hierarchical Clustering
│   │   │   └── Python
│   │   │       ├── Mall_Customers.csv
│   │   │       ├── README.md
│   │   │       └── hierarchical_clustering.ipynb
│   │   └── K-Means Clustering
│   │       └── Python
│   │           ├── Mall_Customers.csv
│   │           ├── README.md
│   │           └── k_means_clustering.ipynb
│   ├── Data Preprocessing
│   │   └── Data Preprocessing
│   │       └── Python
│   │           ├── Data.csv
│   │           ├── README.md
│   │           ├── data_preprocessing_template.ipynb
│   │           └── data_preprocessing_tools.ipynb
│   ├── Deep Learning
│   │   ├── Artificial Neural Networks (ANN)
│   │   │   ├── Python
│   │   │   │   ├── Churn_Modelling.csv
│   │   │   │   ├── README.md
│   │   │   │   └── artificial_neural_network.ipynb
│   │   │   └── Stochastic_Gradient_Descent.png
│   │   └── Convolutional Neural Networks (CNN)
│   │       └── Python
│   │           ├── README.md
│   │           └── convolutional_neural_network.ipynb
│   ├── Dimensionality Reduction
│   │   ├── Kernel PCA
│   │   │   └── Python
│   │   │       ├── README.md
│   │   │       ├── Wine.csv
│   │   │       └── kernel_pca.ipynb
│   │   ├── Linear Discriminant Analysis (LDA)
│   │   │   └── Python
│   │   │       ├── README.md
│   │   │       ├── Wine.csv
│   │   │       └── linear_discriminant_analysis.ipynb
│   │   └── Principal Component Analysis (PCA)
│   │       └── Python
│   │           ├── README.md
│   │           ├── Wine.csv
│   │           └── principal_component_analysis.ipynb
│   ├── Machine Learning(Model Selection)
│   │   ├── Classification
│   │   │   ├── Data.csv
│   │   │   ├── README.md
│   │   │   ├── decision_tree_classification.ipynb
│   │   │   ├── k_nearest_neighbors.ipynb
│   │   │   ├── kernel_svm.ipynb
│   │   │   ├── logistic_regression.ipynb
│   │   │   ├── naive_bayes.ipynb
│   │   │   ├── random_forest_classification.ipynb
│   │   │   └── support_vector_machine.ipynb
│   │   ├── Machine Learning(Model Selection).code-workspace
│   │   └── Regression
│   │       ├── Data.csv
│   │       ├── README.md
│   │       ├── decision_tree_regression.ipynb
│   │       ├── multiple_linear_regression.ipynb
│   │       ├── polynomial_regression.ipynb
│   │       ├── random_forest_regression.ipynb
│   │       └── support_vector_regression.ipynb
│   ├── Model Selection and Boosting
│   │   ├── Model Selection
│   │   │   └── Python
│   │   │       ├── README.md
│   │   │       ├── Social_Network_Ads.csv
│   │   │       ├── grid_search.ipynb
│   │   │       └── k_fold_cross_validation.ipynb
│   │   └── XGBoost
│   │       └── Python
│   │           ├── Data.csv
│   │           ├── README.md
│   │           └── xg_boost.ipynb
│   ├── Natural Language Processing
│   │   └── Natural Language Processing
│   │       └── Python
│   │           ├── README.md
│   │           ├── Restaurant_Reviews.tsv
│   │           └── natural_language_processing.ipynb
│   ├── Regression
│   │   ├── Decision Tree Regression
│   │   │   └── Python
│   │   │       ├── Position_Salaries.csv
│   │   │       ├── README.md
│   │   │       └── decision_tree_regression.ipynb
│   │   ├── Multiple Linear Regression
│   │   │   └── Python
│   │   │       ├── 50_Startups.csv
│   │   │       ├── README.md
│   │   │       └── multiple_linear_regression.ipynb
│   │   ├── Polynomial Regression
│   │   │   └── Python
│   │   │       ├── Position_Salaries.csv
│   │   │       ├── README.md
│   │   │       └── polynomial_regression.ipynb
│   │   ├── Random Forest Regression
│   │   │   └── Python
│   │   │       ├── Position_Salaries.csv
│   │   │       ├── README.md
│   │   │       └── random_forest_regression.ipynb
│   │   ├── Simple Linear Regression
│   │   │   └── Python
│   │   │       ├── README.md
│   │   │       ├── Salary_Data.csv
│   │   │       └── simple_linear_regression.ipynb
│   │   └── Support Vector Regression (SVR)
│   │       └── Python
│   │           ├── Position_Salaries.csv
│   │           ├── README.md
│   │           └── support_vector_regression.ipynb
│   └── Reinforcement Learning
│       ├── Thompson Sampling
│       │   ├── Python
│       │   │   ├── Ads_CTR_Optimisation.csv
│       │   │   ├── README.md
│       │   │   └── thompson_sampling.ipynb
│       │   └── Thompson_Sampling_Slide.png
│       └── Upper Confidence Bound (UCB)
│           ├── Python
│           │   ├── Ads_CTR_Optimisation.csv
│           │   ├── README.md
│           │   └── upper_confidence_bound.ipynb
│           └── UCB_Algorithm_Slide.png
└── README.md
```


## Running Notebooks on Google Colab

1. Open the Jupyter notebook (.ipynb) file on GitHub.
2. Click on the "Open in Colab" button at the top of the notebook.
3. Run the notebook cells step-by-step in Google Colab.

## Details of Each Module

### Association Rule Learning
- **Apriori**: Implementation of the Apriori algorithm for market basket analysis.
- **Eclat**: Implementation of the Eclat algorithm for market basket analysis.

### Classification
- **Decision Tree Classification**: Implementation of decision tree classification.
- **K-Nearest Neighbors (K-NN)**: Implementation of K-NN algorithm.
- **Kernel SVM**: Implementation of Kernel Support Vector Machine.
- **Logistic Regression**: Implementation of logistic regression.
- **Naive Bayes**: Implementation of Naive Bayes classifier.
- **Random Forest Classification**: Implementation of random forest classification.
- **Support Vector Machine (SVM)**: Implementation of Support Vector Machine.

### Clustering
- **Hierarchical Clustering**: Implementation of hierarchical clustering.
- **K-Means Clustering**: Implementation of K-means clustering.

### Data Preprocessing
- **Data Preprocessing**: Techniques for preprocessing data.

### Deep Learning
- **Artificial Neural Networks (ANN)**: Implementation of ANN.
- **Convolutional Neural Networks (CNN)**: Implementation of CNN.

### Dimensionality Reduction
- **Kernel PCA**: Implementation of Kernel Principal Component Analysis.
- **Linear Discriminant Analysis (LDA)**: Implementation of Linear Discriminant Analysis.
- **Principal Component Analysis (PCA)**: Implementation of Principal Component Analysis.

### Model Selection and Boosting
- **Model Selection**: Techniques for model selection including grid search and k-fold cross-validation.
- **XGBoost**: Implementation of XGBoost algorithm.

### Natural Language Processing
- **Natural Language Processing**: Techniques for processing natural language data.

### Regression
- **Decision Tree Regression**: Implementation of decision tree regression.
- **Multiple Linear Regression**: Implementation of multiple linear regression.
- **Polynomial Regression**: Implementation of polynomial regression.
- **Random Forest Regression**: Implementation of random forest regression.
- **Simple Linear Regression**: Implementation of simple linear regression.
- **Support Vector Regression (SVR)**: Implementation of support vector regression.

### Reinforcement Learning
- **Thompson Sampling**: Implementation of Thompson Sampling algorithm.
- **Upper Confidence Bound (UCB)**: Implementation of UCB algorithm.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any changes or improvements.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
