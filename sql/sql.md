# What is the impact of the HbA1c Measurement on Hospital Readmission Rates and Overall Safety? 

## Introduction

The management of blood sugar levels in diabetics has a **significant impact on the control of diabetes and mortality rates**. In 2014, a database with over 74 million hospital encounters yielded over 70,000 encounters of patients who were admitted and identified as diabetics. These data were collected for assessment to understand how to better serve diabetic patients and **increase patient safety**. 


My administrator is trying to understand the current state of a large hospital system. She is swamped and needs some information to quickly make some decisions.  The data is comprised of 10 years worth of clinical care data for over 130 hospitals and integrated networks. I have been tasked with answering several questions. 

I used SQL to answer each question. 

## Objectives

I am assuming the role of data analyst for the hospital. I have been tasked with answering specific questions by the hospital administrator to improve patient safety in the hospital system for diabetics. Specifically, I am to address the following concerns:
1.	Treatment bias by race: find the average number of lab procedures by race. 
Does it appear that different races are being treated differently?
2.	Number of lab procedures as related to time spent in the hospital: categorize the number of lab procedures as few, average and many
Does there appear to be a relationship between the number of lab procedures and the amount of time spent in the hospital?

3.	List the medical specialties that have an average number of procedure count above 2.5 with the total procedure count above 50. Are there some specialties that are performing more procedures than expected?

4.	List all patients who had an emergency but left the hospital faster than the average.
  Is there something different about these patients? Are they part of a specific group?
5.	Research needs a list of all patient numbers who are African-America or have a "Up" to metformin
   Does this group have any noteworthy characteristics?
6.	Identify the distribution of time spent in the hospital
   Is the distribution normal or is there some other distribution that better describes the data?
   
7.	Hospital stays by duration
   Do the majority of patients stay less than 7 days? Once patients stay over 7 days, are these patients very acute?

## Key Findings





## SQL Commands Used in this Project

WHERE | FROM | GROUP BY | DISTINCT | ORDER BY | HAVING | COUNT | SUM | AVG | MAX 

I used https://csvfiddle.io/ to upload a csv file and convert it to a SQL file. This program allows for easy, 100% in-browser querying using SQL commands. 

## EDA

There were 101,766 records in the dataset; of those, 71,518 were unique patients.

There were 5 identified races in the data set: Caucasian (76,099), African American (19,210), Asian (641), Hispanic (2,037) and Other (1,506). There were 2,273 records with unindentified race.

## Analysis

Average number of lab procedures by race: 

|Race | Avg No. of Lab Procedures| Avg Time in Hospital|
------|---------------------------|---|
|Unidentified|	44.104707435107784	|4.289485261768588|
|AfricanAmerican	|44.08505986465383|	4.507860489328475
|Other|	43.43691899070385|	4.273572377158034|
|Caucasian|	42.83294130014849|	4.385721231553634|
|Hispanic|	42.793323514973|	4.059891998036328|
|Asian|	41.21216848673947|	3.995319812792512|

Clearly, the unidentified group and the African American groups had more procedures than any other group, but those same groups did 















