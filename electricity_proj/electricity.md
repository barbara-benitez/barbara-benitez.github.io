# Electricity Price Prediction Project
### Can we predict the price of electricity given just a few indicators?

Upmath extremely simplifies this task by using Markdown and LaTeX. It converts the Markdown syntax extended with LaTeX equations support into HTML code you can publish anywhere on the web.

[Custom foo description](#foo)


 links: [service main page](/ "link title");
 

1. [The Problem and strategy](#problem)
2. [Exploratory data analysis](#Exploratory-Data-Analysis)
3. Preprocessing the data(#Preprocessing-the-Data)
4. Machine learning models
5. Final conclusion
 
 
 
 
 
 # Problem
 ## Pricing Electricity: Challenge and Strategy
 
Predicting the price of electricity is a multifaceted problem that influenced by a myriad of physical and economic factors. 
The challenge I have been given is to forecast the price of electricity for a company, given a dataset including 16 
factors and the historical price of electricity for over 38,000 instances.

I am free to select any modeling algorithm to make the forecasts, but I need to justify the final model that I select to make predictions. 

In the exploration, I explored four supervised machine learning algorithms: linear regression, linear regression with VIF applied, support vector machines, and 
gradient boost regression. Support vector machines and gradient boost gave similarly precise models. I would recommend the gradient 
booster regression model over the support vector machines as it is not as demanding of computational resources.
 
 ## Glossary of terms used 
 
DateTime: Date and time of the record
Holiday: contains the name of the holiday if the day is a national holiday
HolidayFlag: contains 1 if it’s a bank holiday otherwise 0
DayOfWeek: contains values between 0-6 where 0 is Monday
WeekOfYear: week of the year
Day: Day of the date
Month: Month of the date
Year: Year of the date
PeriodOfDay: half-hour period of the day
ForcastWindProduction: forecasted wind production
SystemLoadEA forecasted national load
SMPEA: forecasted price
ORKTemperature: actual temperature measured
ORKWindspeed: actual windspeed measured
CO2Intensity: actual C02 intensity for the electricity produced
ActualWindProduction: actual wind energy production
SystemLoadEP2: actual national system load
SMPEP2: the actual price of the electricity consumed (labels or values to be predicted)
 
 # Exploratory Data Analysis
 
 # Preprocessing the Data
