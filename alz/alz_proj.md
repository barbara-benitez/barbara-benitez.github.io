


<p align="center"><strong> What Markers Should We Look at to Best Predict the Onset of Alzeimer's Disease?</strong></p>
<p align="center">
<img src="/alz/senior_couple.jpg?raw=true" width="500" height="400" alt="My Image">
<span style="font-size: small; text-align: right;">
</p>

## According to the Alzheimer's Association, as of 2023, the health and long-term cost of people living with Alzheimer's disease and other dementias are projected to be $345 billion dollars. 

My goal with this project was to examine several factors that are often tracked as factors that influence the development of Alzheimer's disease. This is a disease that I know and understand due to personal experience. Any insights that can help us understand which factors are implicated in its manifestation is critical to early detection. Accurately detecting  Alzheimer's can result in additional years of quality of life for those affected by the disease. 

This project focused on two aspects of this inquiry:
   1. Which supervised learning model would provide the best identification of an individual with Alzheimer's disease? 
   2. Which features of the leading model provided the most insight?


## TLDR
I explored the effectiveness of the six models in two different ways. First, I used 5-fold cross validation to evaluate the models with the standard hyperameters for ach alogrithm. In each case, the clinical dementia rating (CDR) was the most important indicator of disease. I tuned the hyperparameters of six supervised learning models and found that gradient boost was the most accurate in predicting Alzheimer's disease in the data. This exploration 


Table of Contents

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

 

# What I Expected Versus What I Got

We know that women tend to have a higher incidence of the disease than men (approximately 66% of women as opposed to 34%  of men) so I expected the gender of the individual to have a greater effect on the diagnosis of women. Current research has implicated factors such as diet and lifestyle interventions to prevent cognitive decline. (See work from Dr. Martha Clare Morris, Rush University Medical Center, Alzheimer's Association, and National Institute on Aging to name a few.) These factors are generally closely linked iwth socioeconomic status and years of education. 

# Getting Started with the Data

# Exploring the Data

# Pre-Processing the Data

# The Models with Standard 




