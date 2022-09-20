import pandas as pd
from preprocessing.feats_engineer import feature_engineer

def data_load():
    # Data load
    train_df = pd.read_csv("tabular-playground-series-sep-2022/train.csv", parse_dates=["date"])
    test_df = pd.read_csv("tabular-playground-series-sep-2022/test.csv", parse_dates=["date"])

    # Preprocessing
    train_nocovid_df = train_df.loc[~((train_df["date"] >= "2020-03-01") & (train_df["date"] < "2020-06-01"))]
    #get the dates to forecast for
    test_total_sales_df = test_df.groupby(["date"])["row_id"].first().reset_index().drop(columns="row_id")
    #keep dates for later
    test_total_sales_dates = test_total_sales_df[["date"]]

    train_total_sales_df = feature_engineer(train_nocovid_df)
    test_total_sales_df = feature_engineer(test_total_sales_df)

    y = train_total_sales_df["num_sold"]
    X = train_total_sales_df.drop(columns="num_sold")
    X_test = test_total_sales_df

    return X, y, X_test
