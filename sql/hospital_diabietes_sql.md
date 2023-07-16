# What is the impact of the HbA1c Measurement on Hospital Readmission Rates and Overall Safety? 

## Introduction

The management of blood sugar levels in diabetics has a **significant impact on the control of diabetes and mortality rates**. In 2014, a database with over 74 million hospital encounters yielded over 70,000 encounters of patients who were admitted and identified as diabetics. These data were collected for assessment to understand how to better serve diabetic patients and **increase patient safety**. 


My administrator is trying to understand the current state of a large hospital system. She is swamped and needs some information to quickly make some decisions.  The data is comprised of 10 years worth of clinical care data for over 130 hospitals and integrated networks. I have been tasked with answering several questions. 

I used SQL to answer provide insight and make actionable recommendations.

## Objectives

I am assuming the role of data analyst for the hospital. I have been asked specific questions by the hospital administrator to improve patient safety in the hospital system for diabetics. Specifically, I am to address the following concerns:
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
1. There were five race groups in the data. The unidentified group and the African American group had the highest number of lab procedures overall.
2. There was definitely a direct relationship between the average number of lab procedures and the amount of time spent in the hospital. In fact, there was roughly a two procedure increase for each additional day the patient stayed in the hospital on average. 

3. There were 6 specialties that were performing more procedures than the specified threshold. None of these specialties seemed unreasonable.
4. It was noteworthy that the largest group to leave the hospital faster than the average was unidentified in terms of specialty. Approximately 50% of these patients were over 50 years old. 
5. There were exactly 206 patients of African American race that were coded with metformin "Up".
6. The distribution of the time spent in the hospital was decidedly right-skewed; that is, most of the patients had short stays with the hospital.
7. The data showed that 79.2% of the patients had stays of less than 7 days. Of the remaining 20.8%, the only 20.5% were acute. It appears that the administration should require additional scrutiny for those patients having longer stays without an acute diagnosis. 
   



## SQL Commands Used in this Project

WHERE | FROM | GROUP BY | DISTINCT | ORDER BY | HAVING | COUNT | SUM | AVG | MAX 

I used https://csvfiddle.io/ to upload a csv file and convert it to a SQL file. This program allows for easy, 100% in-browser querying using SQL commands. 

## EDA

There were 101,766 records in the dataset; of those, 71,518 were unique patients.

There were 5 identified races in the data set: Caucasian (76,099), African American (19,210), Asian (641), Hispanic (2,037) and Other (1,506). There were 2,273 records with unindentified race.

## Analysis

The average number of lab procedures by race is summarized in the table.  

|Race | Avg No. of Lab Procedures| Avg Time in Hospital|
------|---------------------------|---|
|Unidentified|	44.10	|4.29|
|AfricanAmerican	|44.09|	4.51|
|Other|	43.44|	4.27|
|Caucasian|	42.83|	4.39|
|Hispanic|	42.79|	4.06|
|Asian|	41.21|	4.00|

These finding were obtained using the query that follows. 
<p align="center">
  <img src="timeinhopt_labproc.jpg?raw=true"   height="200" alt="My Image">
<span style="font-size: small; text-align: right;">
</p>


Clearly, the unidentified group and the African American groups had more procedures than any other group, but those same groups did 

The number of lab procedures as compared to the number of days in the hospital increased by approximately two procedures on average. 
| No. of Days in Hospital |  Avg no. of Lab Procedures |
--------------------------|-----------|
| Few Days  |
|1|	32.92|
|2|	37.70|
|3|	40.39|
|4|	43.80|
| Average Days  |
|5|	46.75|
|6	|48.50|
|7|	50.12|
|8|	51.42|
|9|	53.27|
|High Days |
|10|	53.57|
|11|	54.26|
|12|	54.89|
|13|	55.60|
|14|	55.59|


There were 6 specialties that exceeded the 2.5 average number of procedures and had over 50 procedures on the books. These specialties were obtained with the query
<p align="center">
  <img src="hosp_specialty_gt_2,5.jpg?raw=true"   height="200" alt="My Image">
<span style="font-size: small; text-align: right;">
</p>

| Specialty | Avg. No of Procedures |
--------|---------|
|Surgery-Thoracic	|3.50|
|Surgery-Cardiovascular/Thoracic	|3.25|
|Radiologist	|3.24|
|Cardiology|	2.70|
|Surgery-Vascular|	2.57|
|Radiology|	2.53|

These all appeared to be very reasonable considering that each surgery was likely for a non-elective, medically necessary procedure. 


Now the administrator also wanted a listing of all of the patients that came in for emergency procedures but left faster than the average patient. This list was generated with the query that follows and resulted in 7677 patients.  
<p align="center">
  <img src="emerg_less_than.jpg?raw=true" alt="My Image">
<span style="font-size: small; text-align: right;">
</p>
It was noteworthy that 3,944 of these emergency patients were admitted without an identified specialty. It would certainly be helpful to understand if these were truly emergencies or perhaps visits that were initially entered as emergencies but were really perhaps routine appointments for individuals without insurance or access to regular medical services. Emergency and trauma was the next largest group with 859 patients. Again, comparing the number in the  unidentified group to the emergency/trauma group was surprising. I would definitely recommend that the hospital require more stringent categorization of specialty as the patients are admitted to help better identify how and whyt their average stay was so much lower. It was also noted that roughly 2,000 of these patients were over the age of 50. 









