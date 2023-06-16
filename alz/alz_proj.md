


<p align="center"><strong> What Markers Should We Look at to Best Predict the Onset of Alzeimer's Disease?</strong></p>
<p align="center">
<img src="/alz/senior_couple.jpg?raw=true" width="500" height="400" alt="My Image">
<span style="font-size: small; text-align: right;">
</p>

## According to the Alzheimer's Association, as of 2023, the health and long-term cost of people living with Alzheimer's disease and other dementias are projected to be $345 billion dollars. 

My goal with this project was to examine several factors that are often tracked as factors that influence the development of Alzheimer's disease. This is a disease that I know and understand due to personal experience. Any insights that can help us understand which factors are implicated in its manifestation can be critical to early detection of the disease. Early detection can result in additional years of quality of life for those affected by the disease. 

This project focused on two aspects of this inquiry:
   1. Which supervised learning model would provide the best identification of an individual with Alzheimer's disease? 
   2. Which features of the leading model provided the most insight?


## TLDR
I explored the effectiveness of the six models in two different ways. First, I used 5-fold cross validation to evaluate each model using its default hyperparameters. In every model, the clinical dementia rating (CDR) was the most important indicator of disease. In the cross validation runs, the Random Forest algorithm outperformed the other models. 

Next I used GridSearch to tune the hyperparameters of each learning model. Gradient Boost was the most accurate in predicting Alzheimer's disease in the data, and the most important feature used in predicting Alzheimer's was again the CDR. 

A clinical dementia interview produces the CDR. Clearly, this interview is a reliable means of identifying the presence of Alzheimer's. I believe that it should be administered as individuals increase in age and risk for Alzheimer. At present, this interview is typically  performed as part of the comprehensive assessment of individuals suspected of having dementia or cognitive impairment.  Given that the disease is believed to unfold years before any manifestation of symptoms, I believe that it would be useful to begin interviews before the onset of symptoms. This measure comes at little cost to the individual and practitioner but can have great implications if it can be used to identify deterioration of the mind earlier. The implications of identifying individuals at risk earlier can have enormous economic and healthcare implications. 


## Table of Contents

1. [The Problem](#Problem-Statement)
2. [What I Expected Versus What I Got](#What-I-Expected-Versus-What-I-Got)
3. [Getting Started with the Data](#Getting-Started-with-the-Data)
4. [Exploring the Data](#Exploring-the-Data)
5. [Pre-processing the Data](#Pre-processing-the-Data)
6. [The Models](#The-Models)
7. [Final Thoughts](#Final-Thoughts)


## Glossary of Terms

Group is the target class for the models

- Group: 
   - Nondemented (not classified as having Alzheimer)
   - Demented (classified as having Alzheimer)
   - Converted (Converted to Demented Group during the study)
- Age: Age of the individual
- EDUC: Number of years of education 
- SES: Socioeconomic status from 1-5 (1 is low, 5 is high)
- MMSE: Mini Mental State Examination 
    - scores based on questionaire
    - highest possible score of 30
    - less than 23 is rated as having cognitive impairment 
- CDR: Clinical Dementia Rating: 
    - no dementia (CDR = 0)
    - questionable dementia (CDR = 0.5)
    - MCI (CDR = 1)
    - moderate cognitive impairment (CDR = 2)
    - severe cognitive impairment (CDR = 3)
- eTIV: Estimated total intracranial volume
- nWBV: Normalized Whole Brain Volume
- ASF: Atlas Scaling Factor (normalization technique used to measure the standardized intercranial volume for comparison)


# Problem Statement 

The classification of Alzheimer's is complex and believed to be affected by both genetic and environmental factors. Moreover, current research indicates that the deterioration of the mind begins 20 or more years before any symptoms of memory loss develop. Hence, the better we can understand the factors that influence this degeneration of the mind, the sooner we can begin to adapt our behavior (lifestyle or clinically) to stave the disease off. 

This exploration w

 

# What I Expected Versus What I Got

Current research has implicated factors such as poor diet and lifestyle as risk factors for cognitive decline. (See work from Dr. Martha Clare Morris, Rush University Medical Center, Alzheimer's Association, and National Institute on Aging to name a few.) Since these factors are generally closely linked with socioeconomic status and years of education, I was certain that they would be highly associated with the prediction of disease. They did not come close, however, to the importance of CDR and age. In fact, socioeconomic status and education were on the lowest end of the factors. 

Another surprising result of the analysis was that the Mini Mental State Examination (MMSE) scored very poorly at predicting the disease. 

# Getting Started with the Data

# Exploring the Data

# Pre-Processing the Data

# The Models

# Final Thoughts
This exploration really made me think. We really need to consider how and when we are diagnosing our aging populations. Currently, the only recommendations we really receive from most doctors is to just be healthy and carry on in the hope that we are not one of the unlucky ones to be diagnosed in the future. That is such a passive way to approach such a debilitating disease. We need to screen well before we have symptoms, and we need to really hone in on those factors most heavily implicated in the development of the disease. 

There are several directions for future exploration.
- Incorporating more data, in particular, data including comorbidity factors such as diabetes and heart disease in order to determine which particular lifestyle factors pose the greatest risk
- Looking at the data using clustering algorithms to see if there are patterns not captured by including the target variable in the data
-  Another exploration would be to look 



