import random

import pandas as pd
from sklearn.neighbors import KDTree

# https://towardsdatascience.com/using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd929b
# ------- data preperation

# stations = pd.read_csv("Sell_df_filtered.csv")
# print(stations.head)

points = pd.read_csv("Rent_S2_Done.csv")
points_orig = pd.read_csv("Rent_S2_Done.csv")
# points = points.drop(["title","r3_double1group_id","deal","selling_price.1","description","kennummer","selling_price","a_zip_2","a_street","c","a_kat_o_2","a_kat_u_2","a_surface_living","a_sur_usa","a_sur_prop","a_rent_extra_m2","a_rent_extra","a_brutm_mon","a_netm_mon","a_bron_mon","a_brutm_m2","a_netm_m2","a_bron_m2","a_vkp_tot","a_vkp_m2","a_nb_rooms","a_floor","a_sicht","a_ofen","a_balkon","a_wintergarten","a_garten","a_gsitz","a_lift","a_warenlift","a_rollst","a_wasch","a_neu_stand","a_minergie","a_baup","a_autoab1","a_autoab2","a_info","g_day","g_fin","g_ins","g_om","bfscode","a_vkptot2","e_corrtype","e_count","geoprecision1","geoprecision2","geoprecision3","year_built","bfscode1","src1","bfscode2","unique","ktnr"], axis = 1)
# print(points)


# ------- start of the knn
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



#looking at spatial distribution, over representation of Zurich
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




# Seperating the different regions in three different datasets

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


result["random_build_year"] = result.apply(random_baup, axis=1)


# write to file
# result.rename(columns={'a_surface_living': 'living_surface', 'a_netm_mon': 'mrent', 'a_sicht': 'view', 'a_ofen': 'oven', 'a_balkon': 'balcony', 'a_baup': 'const_period'}, inplace=True)
result.to_csv("rents_S3_Done.csv")
print(result.head())

# just some doodling

"""gva = pd.read_csv("points-with-region.csv")
gva.drop(gva[gva['region'] < 21].index, inplace = True)
gva.drop(gva[gva['region'] > 21].index, inplace = True)
gva.to_csv("gva_rent.csv")"""
