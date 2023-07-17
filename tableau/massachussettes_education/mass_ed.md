# Analyzing Massachusetts Education Data: Insights for School Improvement

<p align="center">
  <img src="close-up-hands-holding-diplomas-caps.jpg?raw=true" width="400" height = "400" alt="My Image">
<span style="font-size: small; text-align: right;">
</p>

## Introduction

As a data scientist recently hired by the State of Massachusetts, my task was to analyze the education data obtained from the Massachusetts Department of Education website. The Department of Education Superintendent sought a report to present to the school board, highlighting the state of the school system and addressing key concerns such as the impact of class size on college attendance, identifying top math schools, and pinpointing schools that are struggling the most. 

The dataset used for this analysis was sourced from Kaggle and encompassed milestone data from over 1,800 schools and more than 950,000 students, specifically from the year 2017.  I will present the key findings and recommendations based on my analysis.

## Concerns about Graduation Rates

Understanding graduation rates is crucial as they serve as an essential metric for assessing the effectiveness of the education system. Low graduation rates can indicate various issues within the school system, such as inadequate resources, ineffective teaching methods, or lack of support for struggling students. As an educator with several years of experience, I understand that graduation rates have a profound impact on students' future success and opportunities. 

Identifying schools with low graduation rates helps focus attention on those areas requiring immediate intervention and resources. By addressing the factors contributing to low graduation rates, we can work towards providing every student with a fair chance at academic success and future opportunities. 

## Key Findings and Recommendations



Using visualizations created with Tableau Public, I presented an overview of the state's education system. The report included statistics such as enrollment numbers, graduation rates, and college attendance rates. By analyzing these metrics, the school board gained valuable insights into the overall performance of the system, enabling them to make informed decisions and prioritize areas for improvement.

1. Impact of Class Size on College Attendance:
- Surprisingly, my analysis revealed an interesting relationship between class size and college attendance. Contrary to common assumptions, an increase in average class size seemed to positively correlate with a higher percentage of students attending college.
- It was observed that smaller class sizes (less than 10 students) had significantly lower college attendance rates, regardless of economic advantage or disadvantage.
  I recommend further investigation  to explore potential reasons for this, such as geographic location or specific student needs. Understanding these factors will help design targeted interventions to improve college attendance rates for students in smaller class sizes.

2. Identifying Top Math Schools:
Success in mathematics is often an indicator of logical reasoning skills that translate into post-secondary education success.
- Utilizing data on the 4th-grade math pass rates, I identified the top-performing math schools in the state. I recommend that these schools be used as benchmarks for pedagogical change.
Top performing schools can provide valuable insights for improving math education throughout the district.
I recommend observing practices and sharing them across the system  to enhance performance in mathematics across all schools.

3. Struggling Schools and Recommendations:
To identify struggling schools, I focused on those with a graduation rate below 50%. By isolating and visualizing these underperforming schools in a bar graph, I highlighted the schools having an urgent need for targeted support and interventions.
- I recommend that these struggling schools be provided with additional resources and follow-up support throughout the students' secondary education.
By addressing the challenges faced by these schools, we can work towards increasing overall graduation rates and ensuring every student has equal access to a quality education.


## Dashboard
 The snapshot that follows is of a dashboard that I devised in Tableau Public to illustrate key findings from the data. 
<p align="center">
  <img src="overall_tableau_mass.jpg?raw=true"  alt="My Image">
<span style="font-size: small; text-align: right;">
</p>

For access to the dashboard, please visit https://public.tableau.com/app/profile/barbara.benitez4236/viz/MassechusettesEducationDataSet/MassachusettsEducationSummaryofMilestones.


## Data Analysis

### College Attendance and Class Size 

<p align="center">
  <img src="attend_v_class_size.jpg?raw=true"  alt="My Image">
<span style="font-size: small; text-align: right;">
</p>

Analyzing college attendance and class size is crucial in understanding their impact on graduation rates and overall student success. College attendance is a significant indicator of students' preparedness for higher education and their ability to transition successfully into post-secondary institutions. By examining the relationship between class size and college attendance, we can identify potential factors that either facilitate or hinder students' access to and preparedness for college. 


Contrary to expectations, the analysis of the data did not support the notion that smaller class sizes would lead to higher graduation rates. In fact, the findings revealed a rather surprising trend: there was a fairly linear increase in the graduation rate as class size increased. This unexpected result challenges the conventional belief that smaller class sizes inherently yield better outcomes. 

The data suggest that other factors, such as pedagogical approaches, teacher effectiveness, or available resources, may have a more significant impact on graduation rates than class size alone. I recommmend further investigation  to better understand these factors and uncover the underlying reasons for this observed pattern. By understanding the nuanced dynamics at play, education policymakers and stakeholders can develop more targeted interventions and strategies to improve graduation rates and enhance student success.

### Top Performing Math Schools
Early math skills have long been recognized as strong predictors of educational success. Research consistently shows that students with strong foundational math skills are more likely to excel academically and have better long-term educational outcomes. 

Since these early math skills  are believed to serve as a crucial foundation for developing critical thinking and problem-solving abilities required for navigating the complexities of higher education, I summarized the performance of school districts as a percentage as shown in the chart that follows.


<p align="center">
  <img src="math_pass.jpg?raw=true"  alt="My Image">
<span style="font-size: small; text-align: right;">
</p>

Performance success was divided into three groups by district based on the pass rate: top performing (performing in the 90th percentile or higher), acceptable (at or above 50th percentile) or unacceptable (under 50 percentile). 
- Top performing: the pedagogy and methodologies employed by these schools should be observed and replicated as much as possible
 I highly recommend using the top-performing schools as benchmarks for implementing pedagogical changes in math education. These schools have demonstrated exemplary performance and can offer valuable insights into effective teaching strategies, curriculum design, and instructional methodologies. By studying and implementing the successful practices observed in these schools, we can enhance math education across the district, leading to improved outcomes and increased graduation rates.
- Unacceptable: For those districts that were underperforming, I am recommending early intervention and adoption of pedagogical changes that have provided success in the top-performing districts.



### Struggling Schools



<p align="center">
  <img src="grad_percent.jpg?raw=true"  alt="My Image">
<span style="font-size: small; text-align: right;">
</p>


By singling out schools with graduation rates below 50%, I successfully identified the struggling institutions in need of immediate attention. Visualizing these underperforming schools through a bar graph highlighted the pressing requirement for targeted support and interventions.

I strongly advocate for allocating additional resources and providing continuous support to these struggling schools throughout the secondary education. By addressing the specific challenges faced by these institutions, I believe that we can make substantial progress in boosting overall graduation rates and ensuring equitable access to a high-quality education for every student.

## Final thoughts
The analysis of Massachusetts' education data provided crucial insights into the state of the school system. By understanding graduation rates, exploring the impact of class size on college attendance, identifying top math schools, and pinpointing struggling schools, we can develop targeted interventions and support systems to improve the overall quality of education. It is vital for the State of Massachusetts to invest in data-driven decision-making and continuously monitor progress to foster an educational environment that empowers all students to succeed.
