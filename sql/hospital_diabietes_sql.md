# What is the impact of the HbA1c Measurement on Hospital Readmission Rates and Overall Safety? 

## Introduction

The management of blood sugar levels in diabetics has a **significant impact on the control of diabetes and mortality rates**. In 2014, a database with over 74 million hospital encounters yielded over 70,000 encounters of patients who were admitted and identified as diabetics. These data were collected for assessment to understand how to better serve diabetic patients and **increase patient safety**. 


My administrator is trying to understand the current state of a large hospital system. She is swamped and needs some information to quickly make some decisions.  The data is comprised of 10 years worth of clinical care data for over 130 hospitals and integrated networks. I have been tasked with answering several questions. 

I used SQL (primarily) and Excel to answer provide insight and make actionable recommendations. SQL commands such as WHERE, FROM, GROUP BY, DISTINCT, ORDER BY, HAVING, COUNT, SUM, AVG, MAX, and LIMIT were utilized. The data was imported into Excel to generate visualizations like tables and histograms.




## Objectives

I am assuming the role of data analyst for the hospital. I have been asked specific questions by the hospital administrator to improve patient safety in the hospital system for diabetics. Specifically, I am to address the following concerns:
1.	Treatment bias by race: find the average number of lab procedures by race.
Does it appear that different races are being treated differently?
2.	Number of lab procedures as related to time spent in the hospital: categorize the number of lab procedures as few, average and many.
Does there appear to be a relationship between the number of lab procedures and the amount of time spent in the hospital?

3.	List the medical specialties that have an average number of procedure count above 2.5 with the total procedure count above 50.
   Are there some specialties that are performing more procedures than expected?

5.	List all patients who had an emergency but left the hospital faster than the average.
  Is there something different about these patients? Are they part of a specific group?
6.	Research needs a list of all patient numbers who are African-America or have a "Up" to metformin.
   Does this group have any noteworthy characteristics?
7.	Identify the distribution of time spent in the hospital.
   Is the distribution normal or is there some other distribution that better describes the data?
   
8.	Hospital stays by duration
   Do the majority of patients stay less than 7 days? Once patients stay over 7 days, are these patients very acute?

## Key Findings
Here are the key findings and insights from the analysis:
1. Treatment bias by race:
- The unidentified and African American race groups had the highest number of lab procedures overall.
- Average number of lab procedures was highest for the African American group, followed by the unidentified group.
2. Number of Lab Procedures and Time Spent in the Hospital:
- There was a direct relationship between the average number of lab procedures and the time spent in the hospital.
- On average, there was roughly a two-procedure increase for each additional day a patient stayed in the hospital.
3. Medical Specialties with Higher Procedure Counts: 
- Six specialties exceeded the specified threshold of an average procedure count above 2.5 and a total procedure count above 50.
- These specialties included Surgery-Thoracic, Surgery-Cardiovascular/Thoracic, Radiologist, Cardiology, Surgery-Vascular, and Radiology.
4. Patients with Emergency Discharge Times Faster than Average:
- A significant number of patients who had an emergency procedure left the hospital faster than the average patient.
- It was noteworthy that a large portion of these patients did not have an identified specialty, raising questions about the nature of their emergencies.
5. Patient Group: African-American or "Up" to Metformin:
- There were 206 patients of African American race who were coded with "Up" to metformin.
- Further analysis of this group's noteworthy characteristics may be necessary.  There were exactly 206 patients of African American race that were coded with metformin "Up".
6. Distribution of Time Spent in the Hospital:
  - The distribution of time spent in the hospital was right-skewed, with the majority of patients having short stays.
- A significant proportion of patients (79.2%) stayed in the hospital for less than seven days.
7. Hospital Stays by Duration:
- Majority of patients stayed less than seven days.
- Patients staying over seven days were not necessarily acute, suggesting the need for additional scrutiny and justification for longer stays.   

These findings provide insights into the relationship between HbA1c measurement, hospital readmission rates, and overall safety. They can assist the hospital administration in making informed decisions to improve patient care and resource allocation.


## SQL Commands Used in this Project

WHERE | FROM | GROUP BY | DISTINCT | ORDER BY | HAVING | COUNT | SUM | AVG | MAX | LIMIT

I used https://csvfiddle.io/ to upload a csv file and convert it to a SQL file. This program allows for easy, 100% in-browser querying using SQL commands. 






## EDA

There were 101,766 records in the dataset; of those, 71,518 were unique patients.

There were 5 identified races in the data set: Caucasian (76,099), African American (19,210), Asian (641), Hispanic (2,037) and Other (1,506). There were 2,273 records with unindentified race.

## Analysis

The average number of lab procedures by race is summarized in the table.  

|Race | Avg No. of Lab Procedures| Avg Time in Hospital|
------|---------------------------|---|
|Unidentified|	44.10	|4.29|
|African American	|44.09|	4.51|
|Other|	43.44|	4.27|
|Caucasian|	42.83|	4.39|
|Hispanic|	42.79|	4.06|
|Asian|	41.21|	4.00|

These finding were obtained using the query that follows. 
<p align="center">
  <img src="timeinhopt_labproc.jpg?raw=true"   height="200" alt="My Image">
<span style="font-size: small; text-align: right;">
</p>


- The unidentified group and the African American groups had more procedures than any other group.
- The African American group had the highest average amount of time in the hospital.


### Lab Procedures Versus Number of Days in the Hospital

<p align="center">
  <img src="no_procedures_by_day.jpg?raw=true"  alt="My Image">
<span style="font-size: small; text-align: right;">
</p>

From the graph we can see that the relationship is fairly linear. 
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


### Are there any specialties that are performing more procedures than others? If so, are they justified?

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

### Did any patients coded as emergency procedures leave faster than the average?
Now the administrator also wanted a listing of all of the patients that came in for emergency procedures but left faster than the average patient. This list was generated with the query that follows and resulted in 7677 patients.  
<p align="center">
  <img src="emerg_less_than.jpg?raw=true" alt="My Image">
<span style="font-size: small; text-align: right;">
</p>

  
It was noteworthy that 3,944 of these emergency patients were admitted without an identified specialty. I would recommend investigation into whether or not these were truly emergencies. Perhaps these visits that were initially entered as emergencies were really  routine appointments for individuals without insurance or access to regular medical services. These are important issues to raise with patients of this type.
  
Emergency and trauma was the next largest group with 859 patients. Again, comparing the number in the  unidentified group to the emergency/trauma group was surprising. I would definitely suggest that the hospital require more stringent categorization of specialty as the patients are admitted to help better identify how and why their average stay was so much lower. It was also noted that roughly 2,000 of these patients were over the age of 50. It could also be helpful to consider if age had any implication on the stay.



  ### How are the number of days in the hospital distributed? 

The number of days spent in the hospital has a large impact on the operations of the hospital (bed usage, staff-to-patient ratios) and a large financial impact on the patients themselves. Ideally, the number of days in the hospital reflects the clinical attention that the patient needs. 

My administrator understands the need to balance patient care and hospital resources. We want to understand if the number of days in the hospital system are as expected and if those  with extended stays are justified.

For this, I queried the database and imported the data into Excel to build a histogram of the number of patients per number of days in the hospital. 
<p align="center">
  <img src="daysinhospoverall.jpg?raw=true" alt="My Image">
<span style="font-size: small; text-align: right;">
</p>
These data clearly showed that the majority (79.2%) of patients stayed in the hospital for fewer than seven days. The data were right-skewed, and the majority of patients had stays of three days  in the hospital. 

An additional query was performed to ascertain whether or not those patients having stays of 7 or more days were acute. Surprisingly, this was not the case. Approximately, 1 in 5 patients was acute. I am suggesting that the administration require additional justification for extended stays when the patients are not acute. 


## Final Remarks

The analysis of HbA1c measurement's impact on hospital readmission rates and overall safety has revealed important insights. Treatment bias by race was observed, with variations in the average number of lab procedures among different racial groups. Additionally, a direct relationship was found between the number of lab procedures and the duration of hospital stays.

Several medical specialties were identified as performing more procedures than expected, but these numbers were found to be reasonable for non-elective, medically necessary surgeries. It was also discovered that a significant number of patients with emergency procedures left the hospital faster than the average patient, warranting further investigation into the nature of their emergencies.

The distribution of time spent in the hospital was right-skewed, indicating that most patients had short stays. However, a proportion of patients stayed for more than seven days without acute conditions, suggesting the need for additional justification for extended stays.

These findings provide valuable insights into the hospital system's current state and highlight areas where improvements can be made to enhance patient safety and care for diabetics. By addressing treatment bias, understanding the relationship between lab procedures and hospital stays, and closely examining emergency cases, hospitals can implement targeted interventions to optimize patient outcomes.

By leveraging SQL and Excel for data analysis, the data analyst played a crucial role in providing actionable recommendations to the hospital administration. The use of these tools and techniques can empower healthcare professionals to make data-driven decisions and enhance patient safety across hospital systems.

Through continuous monitoring, analysis, and improvement, hospitals can better serve diabetic patients, optimize resource utilization, and ultimately contribute to improved healthcare outcomes for all.




