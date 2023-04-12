---
output:
  html_document: default
  html_notebook: 
    toc: yes
  toc: true
  pdf_document: default
editor_options: 
  chunk_output_type: inline
---
# Data Preparation

# Table of contents
1. [Introduction](#introduction)
2. [Data Filtering](#paragraph1)
3. [Outlier Analysis](#paragraph2)
4. [Complementing Data](#paragraph3)
5. [Data Bias](#paragraph4)


### Introduction <a name="introduction"></a>
The data provided records all the listings on ImmoScout from 2004 to 2015. Since platform users entered these listings manually, they contain many incorrect and missing values. 

All codes in this annex can offer additional insights into the data, given that they offer interactive graphs and maps. The most useful interactive visualizations can be found under the "Quick Access" folder in this thesis's GitHub repository. Several adjustments have been made to allow a pdf export of this documentation; these will be mentioned. Additionally, the undocumented code used for this thesis is available under the "raw code" folder of the GitHub repository.




```python
#Importing used libraries
import csv
import pandas as pd
import sklearn as sk
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression 
import statsmodels.graphics.api as smg
import plotly.express as px
from datetime import datetime
from scipy.spatial.distance import mahalanobis 
import folium
from folium import plugins
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from sklearn.neighbors import KDTree
import plotly.io as pio
from IPython.display import Image
```

### Data Filtering <a name="paragraph1"></a>


First the data is eyplored in general. Here the file that is loaded is not the original file but already has certain values filtered out (that will be explained in further steps). This was  done as the original file was too large to be handled by Jupyter Notebook (which was used for the creation of this documentation but not for the analysis itself).




```python
df = pd.read_csv('Step_01.csv', nrows=100000)

print('Shape:', df.shape)
print('Information:', df.info())
```

There are 51 variables with 2’225’232 lines. The dataset is now checked for missing values.

```python
df.isna()
```

As shown, there are multiple “True” meaning there is a certain number of missing values. Given a large number of observations, observations with missing values are removed. Exploring the raw data in Excel also showed that many prices (rents) were incorrectly entered in formats such as 10,000 or 10.000.00 instead of 10'000.
The clean_price function removes non-numeric or non-decimal point characters from the string. Afterward, it removes the trailing decimal point if present. If the price is invalid, the function returns an empty string. Otherwise, the function returns the cleaned price. Finally, it checks if prices are correct by limiting the number of digits to 12. The function also checks that a comma does not separate the prices. This is necessary as the original was not a CSV format.


```python
def clean_price(price):
    price = re.sub(r"[^0-9.]", "", price or "").rstrip(".")

    # check whether price is valid
    if len(price) > 12 or price.count(".") > 1:
        return ""

    return price


def clean_row(row):
    row["selling_price"] = clean_price(row["selling_price"])
    row["a_netm_mon"] = clean_price(row["a_netm_mon"])
    return row
```

The following snippet fulfills three functions. It applies the functions defined above to remove rows that contain invalid data. However, only the columns of interest are kept. The original dataset contained 52 variables; however, only ten were kept for two reasons. First, some of these variables were not of interest. The text description, for example, could not be integrated into a regression analysis (NLP was considered but seemed out of the scope of this thesis). Secondly, certain variables had too many missing variables, which would have considerably reduced the number of observations. Variables such as the street name were expluded as they are give the same information as the geographical coordinates. Variables with too many missing values such as if the dwelling has a garden, minergie, washmachine, floor number are removed.

Finally, this script only adds rental offers to the new file “Rent\_S1\_Done.csv”. The variables of interest kept in the final CSV file are listed under the _usecols_ parameter. 



```python
with open("adScanFull.csv") as readfile:  # Name of the original file to filter
    with open("Step_02.csv", "w") as csvfile:  # Name of the new file name.
        reader = csv.DictReader(readfile, delimiter="#")
        writer = csv.DictWriter(
            csvfile, fieldnames=reader.fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for num, row in enumerate(reader):
            if num % 10000 == 0:
                print(f"{num} lines processed.")
            row = clean_row(row)
            if (
                row["deal"]
                and row["a_netm_mon"]
                and row["a_surface_living"]
                and row["a_nb_rooms"]
                and row["a_sicht"]
                and row["a_ofen"]
                and row["a_balkon"]
                and row["a_baup"]
                and row["g_day"]
                and row["longitude2"]
                and row["latitude2"]
            ):
                # Variables which can't have NaN valuies to be included in the
                # new dataframe.
                writer.writerow(row)
                # Above this line are the variables for which, if there is nop
                # value, the row of data (offer) is not included in the
                # data frame.

df_rent = pd.read_csv(
    "Step_02.csv",
    usecols=[
        "deal",
        "a_netm_mon",
        "a_surface_living",
        "a_nb_rooms",
        "a_sicht",
        "a_ofen",
        "a_balkon",
        "a_baup",
        "g_day",
        "longitude2",
        "latitude2",
    ],
    low_memory=False,
)
# check the types
print(df_rent.dtypes)

options = [
    ("RENT")
]  # Above this line are the variables which are included in the new dataframe.
df_rent = df_rent.loc[df_rent["deal"].isin(options)]
df_rent.to_csv("Rent_S1_Done.csv")
```

As mentioned previously, the original file was not loaded on JupyterNotebook as it was too large. The code was run on a different platform. The resulting file was uploaded to JupyterNotebook. Bellow, the resulting file is explored. It shows the list of variables. The data types are consistent, and there are no null values.



```python
df = pd.read_csv('Rent_S1_Done.csv', nrows=100000)
print('Shape:', df.shape)
print('Information:', df.info())
```

### Outliers analysis <a name="paragraph2"></a>



In this following chapter, a series of outlier detection methods are performed on the data. Since the data was most probably manually entered by individual sellers (on the Immoscout website since it is a peer-to-peer platform), the chances of having outliers or wrong values are quite high. 
Additionally, when navigating the ImmoScout24 platform, many sellers indicate a rent of zero and in, and the description instructs the viewer to contact the oﬀice for the real rent. They do this to allow the dwelling to be visible to as many people as possible, even if the user filters results by price. There are many deviations like this one; thus, the reason for the following thorough outlier analysis procedure. In the first step, the file is loaded from the previous step, and the dates are parsed so that it is in a standardized Python format.



```python
#date parser setup
dateparse = lambda x: datetime.strptime(x, '%Y%m%d') 
dateparse2 = lambda x: datetime.strptime(x, '%Y')
#reading the file
df = pd.read_csv('Rent_S1_Done.csv', parse_dates=['g_day'],date_parser=dateparse)
```

Colums which are not needed are removed, such as the tyoe of deal, since only rental offeres are selected, the old index.  


```python
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('deal', axis=1)
```

**Net rents**


First, the distribution function and a boxplot of the net rents are observed. 




```python
#Figure 1 
fig = px.histogram(df, x="a_netm_mon", marginal="box")

# Dimensions
width_inches = 8.01
height_inches = 5

#Annotations
fig.update_layout(
    legend=dict(font=dict(size=16)),
    title=dict(text="Distribution of Rent", font=dict(size=20)),
    xaxis=dict(title=dict(text="Rent", font=dict(size=16))),
    yaxis=dict(title=dict(text="Count", font=dict(size=16))),
)

# DPI for printing
dpi = 300

# Save the figure as a PNG image
pio.write_image(fig, "fig_1.png", width=int(width_inches * dpi), height=int(height_inches * dpi))

# Display the saved image in the notebook
Image(filename="fig_1.png")

#To view interactive plot in JupyterNotebook, uncomment the next line. Warning ! large output file, will make jupyternotebook crash if to many are opened simultunously.
#fig.show()
```

As Figure 1 shows there seem to be some negative values for the rents which must be taken out. After taking out the negative samples the new distribution is displayed in Figure 2.




```python
#Figure 2
fig = px.histogram(df, x="a_netm_mon", marginal="box")

# Dimensions
width_inches = 8.01
height_inches = 5

# DPI for printing
dpi = 300

#Annotations
fig.update_layout(
    legend=dict(font=dict(size=16)),
    title=dict(text="Distribution of Rent", font=dict(size=20)),
    xaxis=dict(title=dict(text="Rent", font=dict(size=16))),
    yaxis=dict(title=dict(text="Count", font=dict(size=16))),
)

# Save the figure as a PNG image
pio.write_image(fig, "fig_2.png", width=int(width_inches * dpi), height=int(height_inches * dpi))

# Display the saved image in the notebook
Image(filename="fig_2.png")

#To view interactive plot in JupyterNotebook, uncomment the next line. Warning ! large output file, will make jupyternotebook crash if to many are opened simultunously.
#fig.show()
```

There are still  many results that lie on the extremes of the distribution curve (and outside of the first and last quartile). 
The upper fence is at 31.52, while the most extreme to the right value is at 450. These values could be very prestigious goods, but the fact that there were only ten transactions at over 171.42 CHF per sqm makes us think they are negligible in the illustration of preferences. On the other hand, one could argue that the data set is biased given its nature (peer-to-peer platform) and that these values shout be oversampled. For this thesis, they will not be considered. 
However, extreme values are not necessarily outliers. To verify this within the context of this study, it would be wise to evaluate the rent per sqm; as the living surface is the primary driver of price, it makes more sense to look at the rent in relation to at least one other attribute. As figure 3 shows, there are still plenty of values outside of the lower and upper fence. These values will also be evaluated again in the Mahala Nobis distance test.




```python
#Adding price per square meter to evaluate ratio realism
df['psqm'] = df.apply(lambda row: row.a_netm_mon / row.a_surface_living, axis=1)

#Defining the upper and lower maximum value of price per square meter
df = df[df.psqm < 170]

#Figure 3
fig = px.histogram(df, x="psqm", marginal="box")

# Dimensions
width_inches = 8.01
height_inches = 5

#Annotations
fig.update_layout(
    legend=dict(font=dict(size=16)),
    title=dict(text="Distribution of rent per Square Meter", font=dict(size=20)),
    xaxis=dict(title=dict(text="Rent per Square Meter", font=dict(size=16))),
    yaxis=dict(title=dict(text="Count", font=dict(size=16))),
)

# DPI for printing
dpi = 300

# Save the figure as a PNG image
pio.write_image(fig, "fig_3.png", width=int(width_inches * dpi), height=int(height_inches * dpi))

# Display the saved image in the notebook
Image(filename="fig_3.png")

#To view interactive plot in JupyterNotebook, uncomment the next line. Warning ! large output file, will make jupyternotebook crash if to many are opened simultunously.#fig.show()
```

The extreme values will also be evaluated again in the Mahala Nobis distance test.

**Build Year**


The following variable is the year the good was built. One challenge was the hidden NaN values under the year "9999" These had to be removed again, reducing the sample size. This is clear in figure 4, where we can see many observations at 9999. There are shy of 700'000 data samples (for a total of 1.3 million), with the building year set to 9999. In simple OLS regressions, the model was more precise with a larger data sample than without the build year as a variable. 


```python

#Figure 4
fig = px.histogram(df, x="a_baup", marginal="box")

# Dimensions
width_inches = 8.01
height_inches = 5

#Annotations
fig.update_layout(
    legend=dict(font=dict(size=16)),
    title=dict(text="Distribution of year of construction", font=dict(size=20)),
    xaxis=dict(title=dict(text="Year of construction", font=dict(size=16))),
    yaxis=dict(title=dict(text="Count", font=dict(size=16))),
)

# DPI for printing
dpi = 300

# Save the figure as a PNG image
pio.write_image(fig, "fig_4.png", width=int(width_inches * dpi), height=int(height_inches * dpi))

# Display the saved image in the notebook
Image(filename="fig_4.png")

#To view interactive plot in JupyterNotebook, uncomment the next line.
#fig.show()
```

Additionally, the building years are not a continuous variable but a categorical one. This would become a problem in the regression as the large number of categories would be diﬀicult to analyze. The categories are visible in table 1.


```python
#Table 1

a_baup = {1400:'1400-1799', 1800:'1800-1899', 1900:'1900-1924', 1925: '1925-1949', 1950:'1950-1959', 1960:'1960-1969', 1970:'1970-1979', 1980: '1980-1989', 1990:'1990-1994', 1995:'1995-1999', 2000:'2000-2004', 2005: '2005-2010'}

print("+------+-------------+")
print("| Code |   Period    |")
print("+------+-------------+")
for code, period in a_baup.items(): print(f"| {code:<4} | {period:<11} |")
print("+------+-------------+")
```

Thus the variable was transformed to a continuous variable. Given the large sameple size, randomly assigning specific dates to observations (within their specified periode) had no effect on an OLS regression model that was used to test if this modification had an effect. This function will only be applied to the data at the end of the outlier analysis to not alter the outputs of further multifactor analysis.


```python
# Add new column for building year, choosing number at random

build_periods = {
    1400: (1400, 1799),
    1800: (1800, 1899),
    1900: (1900, 1924),
    1925: (1925, 1949),
    1950: (1950, 1959),
    1960: (1960, 1969),
    1970: (1970, 1979),
    1980: (1980, 1989),
    1990: (1990, 1994),
    1995: (1995, 1999),
    2000: (2000, 2004),
    2005: (2005, 2010),
}


def random_baup(row):
    a_baup = row["a_baup"]

    if a_baup not in build_periods:
        return a_baup

    begining, end = build_periods[a_baup]
    return random.randrange(begining, end + 1)
```

**Living Surface**


The same procedure as for the rent is carried out for the living surface. Intuitively many of the outliers have already been taken out in the price\-to\-square\-meter analysis.



```python
#Figure 5
fig = px.histogram(df, x="a_surface_living", marginal="box")

# Dimensions
width_inches = 8.01
height_inches = 5

#Annotations
fig.update_layout(
    legend=dict(font=dict(size=16)),
    title=dict(text="Distribution of Living Surface", font=dict(size=20)),
    xaxis=dict(title=dict(text="Living Surface", font=dict(size=16))),
    yaxis=dict(title=dict(text="Count", font=dict(size=16))),
)

# DPI for printing
dpi = 300

# Save the figure as a PNG image
pio.write_image(fig, "fig_5.png", width=int(width_inches * dpi), height=int(height_inches * dpi))

# Display the saved image in the notebook
Image(filename="fig_5.png")

#To view interactive plot in JupyterNotebook, uncomment the next line.
#fig.show()
```

Similarly to the rent, there are still a relatively large number of data points far beyond the upper fence. This however as well does not necessarily mean they are outliers. But looking at these points again from a rent-to-sqm perspective it is clear that there are many data points with prices per sqm of less than 2 CHF. Additionally, on google maps satelite view, many of these extreme data points find themselves in city centers where this price point is very unlikely. Otherwise, some of them were industrial buildings, possibly indicating a warehouse or oﬀice space but unlikely residential housing. After a meticulous case-by-case inspection, all data points with a rent to sqm ratio of less than 8 (14 being the minimum on the 1 October 2022 on Home Gate) were eliminated. The new distribution is show in figure 6.

```python
#Figure 6
fig = px.histogram(df, x="a_surface_living", marginal="box")

# Dimensions
width_inches = 8.01
height_inches = 5

#Annotations
fig.update_layout(
    legend=dict(font=dict(size=16)),
    title=dict(text="Distribution of Living Surface", font=dict(size=20)),
    xaxis=dict(title=dict(text="Living Surface", font=dict(size=16))),
    yaxis=dict(title=dict(text="Count", font=dict(size=16))),
)

# DPI for printing
dpi = 300

# Save the figure as a PNG image
pio.write_image(fig, "fig_6.png", width=int(width_inches * dpi), height=int(height_inches * dpi))

# Display the saved image in the notebook
Image(filename="fig_6.png")

#To view interactive plot in JupyterNotebook, uncomment the next line.
#fig.show()
```

**Number of rooms**


Figure 7 shows a good to have 10 million rooms, which seems unlikely so it is removed.




```python
#Figure 7
fig = px.histogram(df, x="a_nb_rooms", marginal="box")

# Dimensions
width_inches = 8.01
height_inches = 5

#Annotations
fig.update_layout(
    legend=dict(font=dict(size=16)),
    title=dict(text="Distribution of Number of Rooms", font=dict(size=20)),
    xaxis=dict(title=dict(text="Number of Rooms", font=dict(size=16))),
    yaxis=dict(title=dict(text="Count", font=dict(size=16))),
)

# DPI for printing
dpi = 300

# Save the figure as a PNG image
pio.write_image(fig, "fig_7.png", width=int(width_inches * dpi), height=int(height_inches * dpi))

# Display the saved image in the notebook
Image(filename="fig_7.png")

#To view interactive plot in JupyterNotebook, uncomment the next line.
#fig.show()
```

As figure 7 shows, the distribution is now more realistic; there are still about 2000 samples with more than 8.5 rooms. Performing a case-by-case analysis on google maps, most of them are mistakes. However, some seem real, as the satellite view makes big houses visible. As mentioned before, there is a low amount of high-end properties in the data set; thus, eliminating all goods with over 8.5 rooms would make that even worse. Two thousand entries over 1.3 million are not overly significant. Thus they are not removed. Several lines with non-standard room numbers (4.4, 5.7, etc.) are removed.


```python
df = df[df.a_nb_rooms < 30] 
df.drop(df.loc[df['a_nb_rooms']==1.04].index, inplace=True)
df.drop(df.loc[df['a_nb_rooms']==1.07].index, inplace=True) 
df.drop(df.loc[df['a_nb_rooms']== 0.5].index, inplace=True)
```

Figure 8 shows the new distribution after the removal of unconventional room numbers.
```python
#Figure 8
fig = px.histogram(df, x="a_surface_living", marginal="box")

# Dimensions
width_inches = 8.01
height_inches = 5

#Annotations
fig.update_layout(
    legend=dict(font=dict(size=16)),
    title=dict(text="Distribution of Number of Rooms", font=dict(size=20)),
    xaxis=dict(title=dict(text="Number of Rooms", font=dict(size=16))),
    yaxis=dict(title=dict(text="Count", font=dict(size=16))),
)

# DPI for printing
dpi = 300

# Save the figure as a PNG image
pio.write_image(fig, "fig_8.png", width=int(width_inches * dpi), height=int(height_inches * dpi))

# Display the saved image in the notebook
Image(filename="fig_8.png")

#To view interactive plot in JupyterNotebook, uncomment the next line.
#fig.show()
```

**View, Balcony and Region**


The view takes \-1 as a value when it is unknown whether there is a view. Eliminating all data which has \-1 would be quite a significant loss. Thus it is replaced with 0 which stands for no view. It also seems unlikely that an advertiser would forget to say that his property has a nice view. Otherwise, outliers in these categorical variables are diﬀicult to detect with single\-factor methods.



```python
df["a_sicht"] = df["a_sicht"].replace(-1, 0)
```


**Mahala Nobis distance**

Before the Mahala Nobis distance method was applied to the data, several regressions were conducted on the data. The model performed very poorly with RSME values above 2000, and all assumptions of linear regressions were violated. Thus a rather strict threshold of 95 was chosen. Through tedious visual case-by-case inspection of high rents dwellings, most were found to be erroneous. The data is thus biased towards low to medium-high rents but has no rents above 3800, as otherwise, the RSME and linear regression assumptions were violated entirely, and it was difficult to draw significant insights from the data. The Mahala Nobis distance method was performed on the data as some outliers may not seem apparent when observed only in relation to one other variable; however, they do when looking at four simultaneously.
The four most significant variables were taken: Rents, Living surface, age of the building and location.
The basic concept of this method is to analyze the distance of the observation from the central tendency and the covariance between the variables. (Cansiz, 2020)


```python

# Select the variables for testing multivariate outliers
variables = ['a_netm_mon', 'a_surface_living', 'a_baup', 'longitude2', 'latitude2']
data = df[variables]

# Calculate the mean and covariance matrix of the data
mean = data.mean()
covariance = data.cov()

# Calculate the Mahalanobis distance for each data point
distances = []
for index, row in data.iterrows():
    distance = mahalanobis(row, mean, covariance)
    distances.append(distance)

# Set a threshold for outliers based on the chi-squared distribution
threshold = np.percentile(distances, 95)

# Identify outliers as those with a distance greater than the threshold
outliers = []
for i, distance in enumerate(distances):
    if distance > threshold:
        outliers.append(i)

# Remove outliers from the dataframe
df_cleaned = df.drop(df.index[outliers])

# Print the number of outliers removed
#print('Number of outliers removed:', len(outliers))

df = df_cleaned
```

The following distribution resulted of this procedure. 71366 observations were removed at a 95% threshold based on the chi-squared distribution.



```python
#Figure 9
fig = px.histogram(df, x="a_surface_living", marginal="box")

# Dimensions
width_inches = 8.01
height_inches = 5

#Annotations
fig.update_layout(
    legend=dict(font=dict(size=16)),
    title=dict(text="Distribution of Rent", font=dict(size=20)),
    xaxis=dict(title=dict(text="Rent", font=dict(size=16))),
    yaxis=dict(title=dict(text="Count", font=dict(size=16))),
)

# DPI for printing
dpi = 300

# Save the figure as a PNG image
pio.write_image(fig, "fig_9.png", width=int(width_inches * dpi), height=int(height_inches * dpi))

# Display the saved image in the notebook
Image(filename="fig_9.png")

#To view interactive plot in JupyterNotebook, uncomment the next line.
#fig.show()
```

The same simple regression improved the adjusted R square from 0.5551 to 0.5757. Moreover, before this procedure, transforming the living surface as a log did not improve the mode, which intuitively is strange (Belniak & Wieczorek, 2017, p. 65) . After this procedure, transforming the living surface into a log improved the model. The same result is observed for the age of the building when squaring it.


### Complementing Data <a name="paragraph3"></a>
The raw data included the coordinates of each listing. In order to classify each observation to one of the three Swiss regions, a K-Nearest Neighbor algorithm was performed on the data. This algorithm sets different points (reference points), each belonging to a class. The reference points were defined manually, visible in figure 10.



```python
#Figure 10

#Interactive map code --
column_names = ["STATION NAME", "longitude2", "latitude2"]


German = [
("Zurich",8.5391825,47.3686498),#0
("St. Gallen",9.3787173,47.4244818),#1
("Bern",7.4474,46.9480),#2
("Munsingen",7.5628,46.8747),#3
("Thun",7.6280,467580),#4
("Frutigen",7.6469,46.5898),#5
("Wattenwill",7.5098,46.7699),#6
("Wimmis",7.6386,46.6761),#7
("Interlaken",7.8632,46.6863),#8
("Leuk",7.6346,46.3169),#9
("Leukerbad",7.6288,46.3800),#10
("St-niklaus",7.8046,46.1762),#11
("Zermatt",7.4455,46.0111),#12
("Lucerne",8.3093,47.0502),#13
("Bale",7.5886,47.5596),#14
("Coire",9.5320,46.8508),#15
("Aldorf",8.6428,46.8821),#16
("wassen",8.5999,46.7070),#17
("Ilanz",9.2047,46.7742),#18
("Splugen",9.3210,46.5491),#19
("Brig",7.9878,46.3159)#20
    
]
French = [
("Geneva",6.153438,46.201664),#21
("Montreux",6.9106799,46.4312213),#22
("Lausanne",6.6322734,46.5196535),#23
("Aigle",6.9667,46.3167),#24
("Bulle",7.0577268,46.6154512),#24
("Yverdons",6.641183,46.7784736),#25
("Neuchatel",6.931933,46.992979),#26
("La Chaux-de-Fonds",6.8328,47.1035),#27
("Orsières",7.1471,46.0282),#28
("Saignelégier",6.9964,47.2562),#29
("Bassecourt",7.2427,47.3389),#30
("Paverne",6.9406,46.8220)#31
]

Italian = [
("Lugano",8.952130,46.004644),#32
("Locarno",8.795714,46.168683),#33
("Fusio",8.6500,46.4333),#34
("Faido",8.8010,46.4782),#35
("Acquarossa",8.9398,46.4546),#36
("Biasca",8.9705,46.3580),#37
("Cevio",8.6023,46.3177),#38
("Bellinzona",9.0244,46.1946)#39
]


import folium
from folium import plugins
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

m = folium.Map([46.8, 8.33], zoom_start=5)



French = pd.DataFrame(French, columns = column_names)
German = pd.DataFrame(German, columns = column_names)
Italian = pd.DataFrame(Italian, columns = column_names)

French_transac = pd.read_csv("fre_rent.csv") 
Italian_transac = pd.read_csv("ita_rent.csv") 
German_transac = pd.read_csv("ger_rent.csv")



#The K cities
for index, row in French.iterrows():
   folium.CircleMarker([row['latitude2'],row['longitude2']],
                    radius=7,
                    #popup=row['Density'],
                    fill_color="blue", # divvy color
                    fill_opacity=0.5,
                    color = 'blue'
                   ).add_to(m)

for index, row in German.iterrows():
   folium.CircleMarker([row['latitude2'],row['longitude2']],
                    radius=7,
                    #popup=row['Density'],
                    fill_color="green", # divvy color
                    fill_opacity=0.5,
                    color = 'green',
                   ).add_to(m)

for index, row in Italian.iterrows():
   folium.CircleMarker([row['latitude2'],row['longitude2']],
                    radius=7,
                    #popup=row['Density'],
                    fill_color="red", # divvy color
                    fill_opacity=0.5,
                    color = 'red',
                   ).add_to(m)
#Uncomment to see 
#m

#Image map code --

image = mpimg.imread("map_1.png")
plt.figure(figsize=(20, 20))
plt.axis('off')

# Display image
plt.imshow(image)
plt.show()
```

This algorithm is largely based on [the works of Corey Hanson](<https://towardsdatascience.com/using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd >) using skit-learn library.(Hanson, 2020)




```python
points = pd.read_csv("Rent_S2_Done.csv")
points_orig = pd.read_csv("Rent_S2_Done.csv")

# ------- start of the KNN
column_names = ["STATION NAME", "longitude2", "latitude2"]


cities = [
    ("Zurich", 8.5391825, 47.3686498),  # 0
    ("St. Gallen", 9.3787173, 47.4244818),  # 1
    ("Bern", 7.4474, 46.9480),  # 2
    ("Munsingen", 7.5628, 46.8747),  # 3
    ("Thun", 7.6280, 467580),  # 4
    ("Frutigen", 7.6469, 46.5898),  # 5
    ("Wattenwill", 7.5098, 46.7699),  # 6
    ("Wimmis", 7.6386, 46.6761),  # 7
    ("Interlaken", 7.8632, 46.6863),  # 8
    ("Leuk", 7.6346, 46.3169),  # 9
    ("Leukerbad", 7.6288, 46.3800),  # 10
    ("St-niklaus", 7.8046, 46.1762),  # 11
    ("Zermatt", 7.4455, 46.0111),  # 12
    ("Lucerne", 8.3093, 47.0502),  # 13
    ("Bale", 7.5886, 47.5596),  # 14
    ("Coire", 9.5320, 46.8508),  # 15
    ("Aldorf", 8.6428, 46.8821),  # 16
    ("wassen", 8.5999, 46.7070),  # 17
    ("Ilanz", 9.2047, 46.7742),  # 18
    ("Splugen", 9.3210, 46.5491),  # 19
    ("Brig", 7.9878, 46.3159),  # 20
    ("Geneva", 6.153438, 46.201664),  # 21
    ("Montreux", 6.9106799, 46.4312213),  # 22
    ("Lausanne", 6.6322734, 46.5196535),  # 23
    ("Aigle", 6.9667, 46.3167),  # 24
    ("Bulle", 7.0577268, 46.6154512),  # 24
    ("Yverdons", 6.641183, 46.7784736),  # 25
    ("Neuchatel", 6.931933, 46.992979),  # 26
    ("La Chaux-de-Fonds", 6.8328, 47.1035),  # 27
    ("Orsières", 7.1471, 46.0282),  # 28
    ("Saignelégier", 6.9964, 47.2562),  # 29
    ("Bassecourt", 7.2427, 47.3389),  # 30
    ("Paverne", 6.9406, 46.8220),  # 31
    ("Lugano", 8.952130, 46.004644),  # 32
    ("Locarno", 8.795714, 46.168683),  # 33
    ("Fusio", 8.6500, 46.4333),  # 34
    ("Faido", 8.8010, 46.4782),  # 35
    ("Acquarossa", 8.9398, 46.4546),  # 36
    ("Biasca", 8.9705, 46.3580),  # 37
    ("Cevio", 8.6023, 46.3177),  # 38
    ("Bellinzona", 9.0244, 46.1946),  # 39
]
cities = pd.DataFrame(cities, columns=column_names)
# points = pd.DataFrame(points, columns = column_names)

kd = KDTree(cities[["longitude2", "latitude2"]].values, metric="euclidean")
k = 1
distances, indices = kd.query(points[["longitude2", "latitude2"]], k=k)

s = pd.Series([distances, indices])
# s.to_csv("s.csv")

points_categorised = pd.DataFrame(points_orig)
points_categorised_2 = points_categorised.assign(region=indices)
points_categorised_2.to_csv("trash.csv")
# Replacing the numbers with the name of the region

# Seperating the different regions in three different datasets, commented out option to extract csv file, used for further data exploration.
ger = pd.read_csv("trash.csv")
ger.drop(ger[ger["region"] > 20].index, inplace=True)
# ger.to_csv("ger_rent.csv")


fre = pd.read_csv("trash.csv")
fre.drop(fre[fre["region"] <= 20].index, inplace=True)
fre.drop(fre[fre["region"] > 31].index, inplace=True)
# fre.to_csv("fre_rent.csv")

ita = pd.read_csv("trash.csv")
ita.drop(ita[ita["region"] < 32].index, inplace=True)
# ita.to_csv("ita_rent.csv")


ger.region = 0
fre.region = 1
ita.region = 2
frames = [ger, fre, ita]


result = pd.concat(frames)
result = result.drop("Unnamed: 0", axis=1)
result = result.drop("Unnamed: 0.1", axis=1)

```

### Data Bias <a name="paragraph4"></a>

The data is suspected to have two biases, first an overall larger number of observations for the German part of Switzerland.

```python

df = pd.read_csv('rents_S3_Done.csv')

counts = df['region'].value_counts()

# Print the counts for each region category
print(counts)
```

The output is: 
German    114'529
French      6'296
Italian     2'529
As seen, we have much more German data than French or Italian.

Moreover we can speculate that this is due to the popularity of the platform in Zurich as 49.5% of the oberservations are withing that refference point in the KNN.

```python


#Examining at spatial distribution
num_observations_bale = (points_categorised_2['region'] == 14).sum()
num_observations_zurich = (points_categorised_2['region'] == 0).sum()
total_observations = points_categorised_2.shape[0]

percentage_bale = (num_observations_bale / total_observations) * 100
percentage_zurich = (num_observations_zurich / total_observations) * 100

print("Number of observations categorized closest to Bale:", num_observations_bale)
print("Percentage of observations categorized closest to Bale: {:.2f}%".format(percentage_bale))
print("Number of observations categorized closest to Zurich:", num_observations_zurich)
print("Percentage of observations categorized closest to Zurich: {:.2f}%".format(percentage_zurich))
print("Total number of observations:", total_observations)

```

The second bias in the data is the distribution of rents. We see that given the nature of the ImmoScout24 platform, there aren't many "high end" goods. 

```python

df = pd.read_csv("rents_S3_Done.csv")

# Precautionary data filtering
df = df[(df['latitude2'].notnull()) & (df['longitude2'].notnull())]

# Creating dataframe to count number of entries for each location.
grouped = df.groupby(['latitude2', 'longitude2']).size().reset_index(name='count')

# Heatmap using the coordinates and the count of entries as the intensity value
switzerland_map = folium.Map(location=[46.8, 8.3], zoom_start=8)
switzerland_map.add_child(folium.plugins.HeatMap(grouped[['latitude2', 'longitude2', 'count']].values, radius=15))
switzerland_map.save('heatmap.html')
````
The output of this code is visisble in the Quick Access folder in the GitHub repository . We can see that areas such as the Zurich gold coast or the quaie de Cologny in Geneva have close to zero entries.
