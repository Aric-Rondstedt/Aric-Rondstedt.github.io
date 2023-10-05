---
layout:     post
title:      Lab One Regression
subtitle:   The homework about Regression
date:       2023-10-05
author:     世维
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - ML
---
# Lab 1: Regression

> The homework about Regression

## Dataset introduction

乳腺癌数据集
经典的二分类数据集，包含569个样本，每个样本30个特征，阳性样本357，阴性样本212

关于乳腺癌数据集
乳腺癌数据集的原型是一组病灶造影图片，该数据集的提供者在收集到这些图片的初期，首先对图片进行了分析，从每张图片中提取主要的特征，然后编写图片处理程序，从图片中抽取这些特征。本数据集只关注了10个原始特征，然后又求得每个原始特征的标准差和最大值作为两个衍生特征，这样，最终数据集呈现出的效果便是30个特征

## Data preparation

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def data_preparation(df: pd.DataFrame, target_col: str, feature_cols: list, test_size=0.2):
    # Define target variable
    y = df[target_col]
    
    # Feature selection
    X = df[feature_cols]
   
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

```

## Data cleaning 

```python
def data_cleaning(df: pd.DataFrame, y, num_features: list, cat_features: list, fill_strategy='mean'):
    df_cleaned = df.copy()
    
    # 1. Handle missing values at the beginning
    for col in num_features:
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

    # 2. Handle outliers using IQR and ensure y and X have matching rows
    for col in num_features:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filter = (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
        df_cleaned = df_cleaned[filter]
        y = y[filter]
    
    # ... [rest of the data_cleaning function remains unchanged]
    
    return df_cleaned, y
```



## Model construction

```
def construct_model(regularization_type='none', C=1.0):
    if regularization_type == 'none':
        model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000)
    elif regularization_type == 'l1':
        model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
    elif regularization_type == 'l2':
        model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
    elif regularization_type == 'elasticnet':
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000)  # l1_ratio can be adjusted
    else:
        print(f"Regularization type {regularization_type} is not recognized. Using 'none'.")
        model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000)
    
    return model
```

## Training & Test

```
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Assuming data_preparation and data_cleaning functions are already defined
features = data.feature_names.tolist()

# Step 1: Data Preparation
X_train, X_test, y_train, y_test = data_preparation(df, 'target', features)

# Step 2: Data Cleaning
cleaned_X_train, y_train = data_cleaning(X_train, y_train, num_features=features, cat_features=[])
cleaned_X_test, y_test_updated = data_cleaning(X_test, y_test, num_features=features, cat_features=[])
y_pred = model.predict(cleaned_X_test)



# Step 3: Model Construction with L1 regularization as an example
model = construct_model('none', C=1.0)

# Step 4: Train the model
model.fit(cleaned_X_train, y_train)

# Calculate metrics
accuracy = accuracy_score(y_test_updated, y_pred)
recall = recall_score(y_test_updated, y_pred)
precision = precision_score(y_test_updated, y_pred)
f1 = f1_score(y_test_updated, y_pred)
y_prob = model.predict_proba(cleaned_X_test)[:, 1]
roc_auc = roc_auc_score(y_test_updated, y_prob)



print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
```

![截图20231006003611](post-img/截图20231006003611.png)

## Plot the results

```
# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test_updated, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

![截图20231006003635](post-img/截图20231006003635.png)

## Review & Optimize

#### 模型评估

测试集上的结果为模型性能提供了初步估计。准确率、召回率、精确率和F1得分都提供了模型分类效果的不同视角。特别是ROC AUC，它表示模型区分正类和负类的能力，为我们提供了模型性能的综合评估。从已经获得的指标来看，模型显示出了相当不错的性能，但仍有优化的空间。

#### 参数调整

为了进一步提高模型的性能，我们可以考虑进行参数调整。例如，调整正则化参数、正则化类型（L1、L2或ElasticNet）等来查看哪些参数为我们的数据集提供了最好的性能。

#### 交叉验证

尽管单次的训练/测试分割可以提供有关模型性能的信息，但使用K折交叉验证可以为我们提供更稳健的性能评估。它通过多次分割数据集并在每个分割上进行评估，为我们提供了模型性能的平均值和方差。这可以帮助我们了解模型在不同的数据子集上的稳定性，从而更好地评估其泛化能力。

#### 结果总结

整个建模过程旨在建立一个能够准确分类的模型。从已经获得的评估指标来看，模型已经取得了相当不错的效果。模型的优点包括高的准确率和召回率，这意味着它能够准确地识别大多数正类样本。但是，根据ROC AUC的评估，模型在区分正类和负类上还有改进的空间。
