#assignment_question_two.py

import pandas as pd
import matplotlib.pyplot as plt
data_frame = pd.read_csv("crime1.csv")

violent_crimes = data_frame['ViolentCrimesPerPop']

plt.hist(violent_crimes, bins=25)
plt.title("The Distribution of Violent Crimes per Population")
plt.xlabel("Violent Crimes per Population")
plt.ylabel("Frequency")
plt.show()

plt.boxplot(violent_crimes)
plt.title("Boxplot of Violent Crimes per Population")
plt.ylabel("Violent Crimes per Population")
plt.xlabel("Data")
plt.show()

#The histogram shows a right-skewed data. This means there are more locations with less violent
#crimes, and there are not that many areas with high rates of violent crimes.

#The boxplot shows the median as the orange horizontal line. The box represents the 50% of the data.

#There are no outliers as the whiskers of the boxplot extend to the data's minimum and maximum values
#The whiskers are typically calculated as: bottom whisker = Q1 - 1.5*IQR and top whisker = Q3 + 1.5*IQR