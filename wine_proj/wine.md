# Wine Quality Prediction

# Fine wine is like fine linen. <span style="color:blue"> It is difficult to replicate, but you know it when you have it.</span> 

An industrious winemaker has tasked me with the evaluation of wine quality to make it more appealing to discerning palates. The evaluation is to be  based on several measurable characteristics of the wine. The wine drinker then assigns a score to evaluate the quality of the wine. Scores can range from 0 to 8.



This was a particularly fun dataset to examine because the target variable is highly subjective whereas it is coupled with several objective measures in the wine making process. (How acidic do you like your wine?) Moreover, the evaluation of a wine's quality is highly variable even for exceptionally expensive wines. I like a challenge. 



Table of Contents

1. [The Problem](#Problem-Statement)
2. [What I Expected Versus What I Got](#What-I-Expected-Versus-What-I-Got)
3. [Getting Started with the Data](#Getting-Started-with-the-Data)
4. [Exploring the Data](#Exploring-the-Data)
5. [Pre-processing the Data](#Pre-processing-the-Data)
6. [The Models](#The-Models)
7. [Final Thoughts](#Final-Thoughts)

This project will look at a dataset with various characteristics of wine to determine if the wine is of good quality.

## Glossary of Terms

What we are given is:
- fixed acidity 
- volatile acidity     
- citric acid          
- residual sugar       
- chlorides            
- free sulfur dioxide  
- total sulfur dioxide
- density           
- pH                
- sulphates         
- alcohol           
- quality 


# Problem Statement 
You really want to know how good a wine is before it hits the market. Some wines have the advantage of name recognition and garner higher prices just for having a recognizable label, but most wines are left with the hope of dusting the palate with something memorable so that you forget the price tag. That is what a good wine maker wants to replicate. It is just like a fine article of designer clothing that you just have to have. You want a wine that calls to you. Frankly, I want that wine too! (Promontory, Bond, anyone?) It sure would be good to know this before letting the wine rest in a barrel for a few years!

The second issue that there are a lot refinements made to wine to make it palatable. Besides how long it sits in a tank, how much citric acid to add, for instance, changes the acidity of the wine and how ``fresh" it appears to the consumer. There are so many factors that can impact the wine making process so we really want to whittle it down to the most important elements. Boom! Although the prospect of tasting wine with a clipboard in hand sounds fun, it might make for some befuddled, drunken evaluations. We need some machine learning to help us along. Quality control should be enjoyed at the end!

My goal is to determine the most important features, and find a machine learning algorithm that can predict most closely the subjective rating of the wine consumer.

# What I Expected Versus What I Got
I tried several shallow algorithms and one basic neural network in this project.  
<ol>
  <li> Naive Bayes Classifier <br>
I really hoped that this classifier would surprise me. It is used so frequently in the wild for classification problems, and despite the fact that the assumptions governing its use are often not met, it tends to perform relatively well in practice. I knew that there was correlation in the data and that the features were definitely not independent, but I couldn't help myself but explore the model--maybe it would surprise me. Well, it didn't. In fact, it performed the most poorly of all the models.
</li>
<li> Decision Tree Classifier <br>
I knew that this classifier should be able to handle any nonlinear relationships in the data. Also, I was a little concerned about overfitting, but I gave it whirl. It was mediocre at 66% accuracy. I was disappointed, but having more classifiers in my pocket, I forged on.
</li>
<li> KNN Classifier<br>
So this was one that disappointed me. I used KNN with the default 5 neighbors. I really hoped it would uncover some great pattern in the data, but it actually performed worse than the Naive Bayes algorithm.
</li>
<li> Logistic Regression Classifier<br>
I was not too convinced about the linearly separable assumption in the data to be honest. I figured this would probably be a mess, but it turned to be one of the more reliable classifiers. On the one hand, there were not any glaring signs of outliers in the data. Each feature varied pretty smoothly with the quality rating. I admit I was surprised. 
</li>
<li> Ridge Classifier<br>
I expected the Ridge Classifier to model quality relatively well, and it did. It tends to handle collinearity well, and the datset was culled a bit to remove highly correlated features. I used the defaualt tuning of the regularizaion to control overfitting. With another pass at modeling the data, I would probably change alpha to see the model would improve. 
</li>
<li> Random Forest Classifier<br>
Random Forest is just the bomb always as shallow algorithms go. Since the dataset was not that big, computation time was not really an issue. It was the best model for the quality of the wine at approximately 75% correct evaluation. I liked using Random Forest as it was unlikely to overfit even using the default number of trees. 
</li>
<li> XGBoost Classifier<br>
Just as with the Random Forest Classifier I expected it to perform well, and it did. In fact, it was a close second. 
</li>
<li> Neural Network Classifier<br>
</li>
<li> SVM<br>
</li>

It turns out that as explorations go, I expected the models to mostly perform badly, but I also expected the neural network to perform on the high end. 



# Getting Started with the Data

## First, I imported some anticipated dependencies.
### I started with Naive Bayes and the standard scaler since the data were not on a single scale. 
### I'll add dependencies as I run each model. 


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
```

### I read in the csv file which was downloaded. It was originally downloaded from Kaggle. 

```python
df = pd.read_csv("winequality-red.csv")
```
### The data were about as clean as can get so we're set to start playing with the numbers and looking for trends.


# Exploring the Data


### I looked at the head and the tail of the data. The data are comprised of 10 features and the target is variable the wine quality. 

```python
df.head()
```



```python
df.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1594</th>
      <td>6.2</td>
      <td>0.600</td>
      <td>0.08</td>
      <td>2.0</td>
      <td>0.090</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99490</td>
      <td>3.45</td>
      <td>0.58</td>
      <td>10.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1595</th>
      <td>5.9</td>
      <td>0.550</td>
      <td>0.10</td>
      <td>2.2</td>
      <td>0.062</td>
      <td>39.0</td>
      <td>51.0</td>
      <td>0.99512</td>
      <td>3.52</td>
      <td>0.76</td>
      <td>11.2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1596</th>
      <td>6.3</td>
      <td>0.510</td>
      <td>0.13</td>
      <td>2.3</td>
      <td>0.076</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>0.99574</td>
      <td>3.42</td>
      <td>0.75</td>
      <td>11.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1597</th>
      <td>5.9</td>
      <td>0.645</td>
      <td>0.12</td>
      <td>2.0</td>
      <td>0.075</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99547</td>
      <td>3.57</td>
      <td>0.71</td>
      <td>10.2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1598</th>
      <td>6.0</td>
      <td>0.310</td>
      <td>0.47</td>
      <td>3.6</td>
      <td>0.067</td>
      <td>18.0</td>
      <td>42.0</td>
      <td>0.99549</td>
      <td>3.39</td>
      <td>0.66</td>
      <td>11.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


### A quick look at the data showed that
- there were no missing values
- all of the features were coded as float
- the target variable, quality, was coded as integer so it will need to be converted to a categorical variable to indicate high or low quality when the data are pre-processed


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   fixed acidity         1599 non-null   float64
     1   volatile acidity      1599 non-null   float64
     2   citric acid           1599 non-null   float64
     3   residual sugar        1599 non-null   float64
     4   chlorides             1599 non-null   float64
     5   free sulfur dioxide   1599 non-null   float64
     6   total sulfur dioxide  1599 non-null   float64
     7   density               1599 non-null   float64
     8   pH                    1599 non-null   float64
     9   sulphates             1599 non-null   float64
     10  alcohol               1599 non-null   float64
     11  quality               1599 non-null   int64  
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB



```python
np.shape(df)
```

    (1599, 12)


## I took a quick look at the descriptive statistics for the data to get a feel for the means and the spread of the data. 
These data are on very different scales.

```python
df.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Check for missing data


```python
df.isnull().sum()
```




    fixed acidity           0
    volatile acidity        0
    citric acid             0
    residual sugar          0
    chlorides               0
    free sulfur dioxide     0
    total sulfur dioxide    0
    density                 0
    pH                      0
    sulphates               0
    alcohol                 0
    quality                 0
    dtype: int64


### I decided to look at the features alongside the quality rating to see if there were any obvious relationships. 
So I ran a loop to generate 10 plots comparing each feature with quality. 
It was clear that several variables tended to be inverse proportional to the quality whereas some had a direct relationship. If any of the other features shared some obviosu relatipship with the quality it was certainly not transparent to me at this point.

```python
# look at some line plots with the quality rating
# write a loop to generate the 10 plots
cols = (len(df.columns))-1
for col in range(0,cols) :
    
    plt.figure(figsize = (20,6))
    sns.lineplot(x='quality', y=df.columns[col], data = df)
    plt.title(f"{df.columns[col]} versus quality", fontsize = 16)
    plt.show()
```


    
![png](output_12_0.png)
    



    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    



    
![png](output_12_3.png)
    



    
![png](output_12_4.png)
    



    
![png](output_12_5.png)
    



    
![png](output_12_6.png)
    



    
![png](output_12_7.png)
    



    
![png](output_12_8.png)
    



    
![png](output_12_9.png)
    



    
![png](output_12_10.png)
    



```python
# look for correlations in the data
correlations = df.corr(method='pearson')
plt.figure(figsize=(16,12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()
```


    
![png](output_13_0.png)
    



```python
# Iterate through the correlation matrix and print out columns having correlation above .6
print("The following column pairs have a correlation above 0.6:\n")
for col1 in correlations.columns:
    for col2 in correlations.columns:
        if col1 != col2 and correlations.loc[col1, col2] > 0.6:
            print(f"     {col1} and {col2}: {correlations.loc[col1, col2]}\n")
```

    The following column pairs have a correlation above 0.6:
    
         fixed acidity and citric acid: 0.6717034347641065
    
         fixed acidity and density: 0.6680472921189531
    
         citric acid and fixed acidity: 0.6717034347641065
    
         free sulfur dioxide and total sulfur dioxide: 0.6676664504810229
    
         total sulfur dioxide and free sulfur dioxide: 0.6676664504810229
    
         density and fixed acidity: 0.6680472921189531
    


- During the pre-processing phase we will drop the columns for fixed acidity, and free sulfur dioxide due to the high correlation with other features



```python
# Create a clean copy of the data
df_clean = df.copy()
```

## Pre-process the data
- Drop the columns for fixed acidity and free sulfur dioxide
- This will entail rescaling the data using StandardScaler


```python
# drop
df.drop(["fixed acidity", "free sulfur dioxide"], axis=1, inplace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### Since quality is a numeric value, and we are trying to classify the wine as above average or not above average we need to designate a new column for above or below average. We will designate high_quality =1 for wines with a quality at or above 6 or a 0 if it is below the threshold. (Note that the designates high_quality to be in the 50th percentile or higher)


```python
# Create high_quality column based on the condition that it is a 1 if quality 
#scores is greater than 6 using a lambda function
df['high_quality'] = df['quality'].apply(lambda x: 0 if x < 6 else 1)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>high_quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Drop the quality column as it is now encoded in high_quality
df.drop(["quality"],axis = 1, inplace=True)
```

### Set up the feature and target arrays


```python

```


```python
X = df.iloc[:, :-1]
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = df.iloc[:,-1]
y.head()
```




    0    0
    1    0
    2    0
    3    1
    4    0
    Name: high_quality, dtype: int64



## Set up the training and test data


```python
X_test, X_train, y_test, y_train = train_test_split(X, y, random_state =4, test_size =.2)
```


```python

```


```python

```

### Rescale the data using StandardScaler


```python
scaler = StandardScaler().fit(X_train)
```


```python
X_train_scaled = scaler.transform(X_train)
```

## Run the Naive Bayes algorithm 


```python
model = GaussianNB()
```


```python
# Fit the Naive Bayes model to the training data
model.fit(X_train, y_train)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">GaussianNB</label><div class="sk-toggleable__content"><pre>GaussianNB()</pre></div></div></div></div></div>




```python
y_pred = model.predict(X_test)
```


```python
# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Model Accuracy:", accuracy)
```

    Naive Bayes Model Accuracy: 0.4573885848318999


## Decision Tree Classification


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
# create instance of Decision Tree Classifier
dt = DecisionTreeClassifier()
```


```python
# fit the training data
dt.fit(X_train, y_train)
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>




```python
# run the predictions
y_pred_dt = dt.predict(X_test)
```


```python
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
```

    Naive Bayes Model Accuracy: 0.4573885848318999
    Decision Tree Classifier Accuracy Score:  0.6645817044566067


## KNN Classification


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn = KNeighborsClassifier()
```


```python
knn.fit(X_train, y_train)
```




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier()</pre></div></div></div></div></div>




```python
y_pred_knn = knn.predict(X_test)
```


```python
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score", accuracy_knn)
```

    Naive Bayes Model Accuracy: 0.4573885848318999
    Decision Tree Classifier Accuracy Score:  0.6645817044566067
    KNN Classifier Accuracy Score 0.6528537920250196


## Logistic Regression Classification


```python
from sklearn.linear_model import LogisticRegression
```


```python
lg = LogisticRegression(max_iter=1000)
```


```python
lg.fit(X_train, y_train)
```




<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(max_iter=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" checked><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=1000)</pre></div></div></div></div></div>




```python
y_pred_lg = lg.predict(X_test)
```


```python
accuracy_lg = accuracy_score(y_test, y_pred_lg)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
```

    Naive Bayes Model Accuracy: 0.4573885848318999
    Decision Tree Classifier Accuracy Score:  0.6645817044566067
    KNN Classifier Accuracy Score:  0.6528537920250196
    Logistic Regression Accuracy Score:  0.7341673182173573


## Ridge Classification


```python
from sklearn.linear_model import RidgeClassifier
```


```python
ridge = RidgeClassifier()
```


```python
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
```


```python
accuracy_ridge = accuracy_score(y_test, y_pred_ridge)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)
```

    Naive Bayes Model Accuracy: 0.4573885848318999
    Decision Tree Classifier Accuracy Score:  0.6645817044566067
    KNN Classifier Accuracy Score:  0.6528537920250196
    Logistic Regression Accuracy Score:  0.7341673182173573
    Ridge Classifier Accuracy Score:  0.727130570758405


## Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rf = RandomForestClassifier()
```


```python
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```


```python
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)
print("Random Forest Accuracy Score: ", accuracy_rf)
```

    Naive Bayes Model Accuracy: 0.4573885848318999
    Decision Tree Classifier Accuracy Score:  0.6645817044566067
    KNN Classifier Accuracy Score:  0.6528537920250196
    Logistic Regression Accuracy Score:  0.7341673182173573
    Ridge Classifier Accuracy Score:  0.727130570758405
    Random Forest Accuracy Score:  0.7505863956215794


## XGBoost Classifier


```python
from xgboost import XGBClassifier
```


```python
xgb = XGBClassifier()
```


```python
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
```


```python
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)
print("Random Forest Accuracy Score: ", accuracy_rf)
print("XGBoost Classifier Accuracy Score: ", accuracy_xgb)
```

    Naive Bayes Model Accuracy: 0.4573885848318999
    Decision Tree Classifier Accuracy Score:  0.6645817044566067
    KNN Classifier Accuracy Score:  0.6528537920250196
    Logistic Regression Accuracy Score:  0.7341673182173573
    Ridge Classifier Accuracy Score:  0.727130570758405
    Random Forest Accuracy Score:  0.7505863956215794
    XGBoost Classifier Accuracy Score:  0.745113369820172


## Basic Neural Network


```python
from sklearn.neural_network import MLPClassifier
```


```python
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter = 1000, activation='relu', solver='adam', random_state = 12)
```


```python
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
```


```python
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)
print("Random Forest Accuracy Score: ", accuracy_rf)
print("XGBoost Classifier Accuracy Score: ", accuracy_xgb)
print("Neural Network Classifier Accuracy Score: ", accuracy_mlp)
```

    Naive Bayes Model Accuracy: 0.4573885848318999
    Decision Tree Classifier Accuracy Score:  0.6645817044566067
    KNN Classifier Accuracy Score:  0.6528537920250196
    Logistic Regression Accuracy Score:  0.7341673182173573
    Ridge Classifier Accuracy Score:  0.727130570758405
    Random Forest Accuracy Score:  0.7505863956215794
    XGBoost Classifier Accuracy Score:  0.745113369820172
    Neural Network Classifier Accuracy Score:  0.7232212666145426


## SVM Classifier


```python
from sklearn.svm import SVC
```


```python
svc = SVC ()
```


```python
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
```


```python
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)
print("Random Forest Accuracy Score: ", accuracy_rf)
print("XGBoost Classifier Accuracy Score: ", accuracy_xgb)
print("Neural Network Classifier Accuracy Score: ", accuracy_mlp)
print("SVM Classifier Accuracy Score: ", accuracy_svc)
```

    Naive Bayes Model Accuracy: 0.4573885848318999
    Decision Tree Classifier Accuracy Score:  0.6645817044566067
    KNN Classifier Accuracy Score:  0.6528537920250196
    Logistic Regression Accuracy Score:  0.7341673182173573
    Ridge Classifier Accuracy Score:  0.727130570758405
    Random Forest Accuracy Score:  0.7505863956215794
    XGBoost Classifier Accuracy Score:  0.745113369820172
    Neural Network Classifier Accuracy Score:  0.7232212666145426
    SVM Classifier Accuracy Score:  0.6223612197028929



```python

```
