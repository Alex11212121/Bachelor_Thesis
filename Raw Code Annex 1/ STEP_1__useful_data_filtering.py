import csv
import pandas as pd
import re


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


# WARNING REMOVED THE FLOOR NUMBER PUT IN DOCUMENTATION

# For Selling deals !!! LOOK AT ADDITIONAL DATA CLEANING IN OUTLIER V2!!!
with open("adScanFull.csv") as readfile:  # Name of the original file to filter
    with open("Step_01.csv", "w") as csvfile:  # Name of the new file name.
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
                and row["selling_price"]
                and row["a_surface_living"]
                and row["a_nb_rooms"]
                and row["a_floor"]
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

df_sell = pd.read_csv(
    "Step_01.csv",
    usecols=[
        "deal",
        "selling_price",
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
print(df_sell.dtypes)

options = [
    ("SALE")
]  # Above this line are the variables which are included in the new dataframe.


df_sell = df_sell.loc[df_sell["deal"].isin(options)]
df_sell.to_csv("Sell_S1_Done.csv")


# ---------------------------------------- Same thing for Renting Deal----------------------------------------
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
                and row["a_zip_2"]
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
        "a_zip_2",
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
