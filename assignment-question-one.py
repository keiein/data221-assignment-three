#assignment-question-one.py

import pandas as pd
data_frame = pd.read_csv("crime1.csv")

violent_crimes = data_frame['ViolentCrimesPerPop']

mean = violent_crimes.mean()
median = violent_crimes.median()
standard_deviation = violent_crimes.std()
min = violent_crimes.min()
max = violent_crimes.max()
print("Mean: " + str(mean))
print("Median: " + str(median))
print("Standard deviation: " + str(standard_deviation))
print("Minimum: " + str(min))
print("Maximum: " + str(max))

#Comparing mean and median: since the mean is much higher than the median, we can say that it is RIGHT SKEWED.
#There are outliers that are pulling the mean to the higher side.
#The median is unaffected by extreme values since it takes the middle data point of a sorted dataset
#Whereas mean takes the summation of all numerical values and divides it by the number of numeric data
