"""
Q: "Given the data set, do a quick exploratory data analysis to get a feel for the distributions and biases of the data.  
Report any visualizations and findings used and suggest any other impactful business use cases for that data."

A: In order to gain insights from the dataset, a series of histograms were generated for individual parameters: Year, Major, Time, and University. 
These visualizations provide valuable information about the ordering patterns of "FoodX" customers.

The Year_Hist indicates that the primary consumers of "FoodX" are almost exclusivly Sophomores and Juniors. Time_Hist exhibits a bell curve 
shape centered around 12 and 13, with fewer orders during the times of 8, 9, 16, and 17. Univ_Hist reveals that the major consumers of "FoodX"
are from Ball State, Butler, and ISU. A secondary group of moderate consumers come from IUPUI (RIP), Evansville, and ND. Following this, there is a
group with almost no consumers, consisting of Depauw, IU, PU, and Valparaiso. Major_Hist displays a similar stratification as Univ_Hist, with 5 groups 
having high demand, 5 with moderate demand, and 10 others with minimal demand.

It's important to note that this dataset solely represents undergraduate college students. Caution should be exercised when extrapolating these 
findings to other demographics, as non-undergraduate students may exhibit different behaviors and preferences.

One plausible business decision is to apply more marketing to the groups of students who have minimal demand such as those with majors of "Fine Arts", 
"Music", or "Civil Engineering". This idea could further be extended to better target groups from universities or grades that exhibit lower demand. One 
might consider using the lowered demand during the times of 8, 9, 16, 17 as justification to reduce the hours of operation of the food truck, 
however as this is not a entirely representative sample of all consumers this should be advoided without further research.
"""

"""
Q: "Consider implications of data collection, storage, and data biases you would consider relevant here considering Data Ethics, 
Business Outcomes, and Technical Implications"

A: It is critical to recognize that by attempting to use this data to predict a consumers order choice it assumes that their order
is a function of at least the time of day, their major, their grade level and their university. However it is possible that these pieces
of data do not have any true predictive power and any results interpreted from this data would fail to be replicated given another set of 
similar data collected over a different time period. 

The data also has volunteer bias as a consumer is not required to use the app to place a order. So any decisions made from this data only
apply to undergraduate consumers who used the app and filled out the neccisary information.

If 'FoodX' wishes to scale their data collection to be more representative of their consumer base they would first need to consider how they
are going to store the additional data while also ensuring they are sourcing their data in an ethical way. This involves being 
transparent with customers and asking for consent before collecting data and storing data in a secure manor to safe gaurd customers data.

If 'FoodX' scales their data collection they may be able to produce a more generalizable predictive model that can be applyed to more than just 
undergraduate consumers. This will likly boost the overall accuracy and help reduce the number of discounts needed to be given out.
"""


"""
The following produces histograms for the various parameters (Year, Major, Univ, Time) where each parameter has a series of bars 
each representing the freq the different order types apear for the parameter.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('XTern 2024 Artificial Intelegence Data Set.xlsx')

order_types = df["Order"].unique()
cols_to_analyze = list(df.columns)
cols_to_analyze.remove("Order")
bar_width = .1

# Each iteration creates a new graph for Year, Major, Univ, Time
for col in cols_to_analyze:

    ## Find the Frequency that some parameter (Year, Major, ...) results in some order 
    cols_freq = [] #[[int, ...], ...] where each sub array is the column to be displayed, and each int represents the freq of each sub bar
    unique_values = list(df[col].unique()) # ex: [Year 1, Year 2, ...]
    unique_values.sort()

    N = len(unique_values) # number of columns to display in bar graph

    for val in unique_values:
        indexes = df.index[df[col] == val].tolist()
        corresponding_orders = list(df.loc[indexes, "Order"])

        sub_bars = [corresponding_orders.count(order_type) for order_type in order_types]
        cols_freq.append(sub_bars)

    ## Graph the bar chart
    fig, ax = plt.subplots(figsize=(10,7))  

    pos = np.arange(N) # position of the bars on the x-axis

    # Convert into numpy array for easier iterations
    cols_freq = np.array(cols_freq)

    for i in range(len(cols_freq[0])):
        plt.bar(pos, cols_freq[:, i], width=bar_width, edgecolor="black", label=order_types[i])

        # Add text above the bars displaying their freq
        for ind, p in enumerate(pos):
            freq = cols_freq[:, i][ind]
            if freq != 0:
                ax.text(p, freq + 0.05, str(freq), ha='center', va='bottom', fontsize=6, fontweight='bold')

        # Update the position of the bars for the next set of bars
        pos = [p + bar_width for p in pos] 

    plt.xlabel('Parameter', fontweight='bold', fontsize=12) 
    plt.ylabel('Order Type Frequency', fontweight='bold', fontsize=12) 
    
    # Set x-axis ticks and add dividing lines
    ax.set_xticks([p - .55 for p in pos])
    ax.set_xticklabels(unique_values, rotation=10, fontsize = 6)
    for i in range(-1, N):
        ax.axvline((i+.95) * bar_width * len(order_types), color='gray', linestyle='--', linewidth=3)

    plt.legend(fontsize = 6)

plt.show()
