**Week 1 Project: Predicting Customer Churn for Sprint**
For the growth of the company, it is important to keep track on your old customers because if your customers continually leave, it doesn’t speak well for the new customers. This is why in my new role as the Data scientist in charge of predicting customer churn at Sprint, a telecom company, I will be using machine learning to predict the customers that are likely to leave.

**Step 1: Defining the Events**
What is Customer churn? Customer churn or attrition refers to the rate at which customers stop doing business with a company over a certain period of time. It is common within telecommunications, subscription services, and e-commerce as it directly impacts company’s revenue and profitability.
In finding a solution, I want to work on the customer churn for Sprint using data from last 3 months to have a comprehensive data to work with and make precise and informed decision.

**Step 2: Data Collection**
Some of the factors to be considered for data collection to help us make informed decision are monthly home internet usage (in GB), end of a contract or subscription, billing information, dissatisfaction with the service (e.g number of complaints submitted), demography (e.g location), better offers from competitors (in prices and promotions), and changes in personal circumstances (e.g from employed to not employed status).
Using Mockaroo to generate data for the following above, our data was generated in csv. Then we import it into our Jupyter Notebook using the syntax;

```python

import pyforest
data = pd.read_csv(“path.data.csv”)     #import data from a csv file

``` 


**Step 3: Data Preprocessing**
After importing the collected data, we will process the data thoroughly to work on some of the irregularities. These irregularities must be handle to get what we want. The things we want to correct are;
Handling missing data: We will remove rows with missing data, outliers, and duplicate data.
Encode categorical variables: using one-hot encoding to work on data like gender.
Normalizing numerical features like monthly home subscription usage.
Creating a time-based dataset spanning several months to capture evolving customer behaviour.

```python
data.dropna(inplace=True)		#handling missing data
data.drop_duplicates(inplace=True)		#removing duplicate data
data = pd.get_dummies(data, columns=[‘contract_type’, ‘complaints_category’, ‘gender’])	#encoding categorical variables using one-hot encoding
from sklearn.preprocessing import MinMaxScaler
scaler = MixMaxScaler()
data[‘monthly_home_subscription_usage’] = scaler.fit_transform(data[‘monthly_home_subscription_usage’].values.reshape(-1, 1))		#normalize numerical features (e.g monthly home subscription usage)
```
**Step 4: Feature Engineering and Selection**
Now, we need to identify and select critical features for predicting churn with the clean dataset we have been able to generate. Some of the features to be considered are;
The rate of change in monthly home subscription usage
Billing pattern and fluctuations
Complaints submitted
Competitors pricing essentials

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=5)
selected_features = selector.fit_transform(X, y)

from sklearn.model_selection import train_test_split  #split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Step 5: Model Selection**
We will make use of the XGBoost algorithm because it is good for handling imbalanced datasets, capture feature interactions, and feature importance.
import xgboost as xgb

```python
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,  # Number of boosting rounds (trees)
    max_depth=3,       # Maximum depth of each tree
    learning_rate=0.1  # Learning rate
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```
We have to optimize the model’s performance by performing a hyperparameter tuning using the grid search approach.

**Step 6: Model Evaluation**
We will need to check if the model we have deployed is the right one for the predicting churn.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score (y_test, model.predict_proba(X_test)[:, 1])
```
"# project1" 
"# Project1" 
