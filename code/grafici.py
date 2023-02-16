from matplotlib import pyplot as plt


def check_nan_value(dataset):
    # Verifichiamo se ci sono valori NaN
    count_nan = dataset.isnull().sum(axis=0)
    count_nans_filtered = count_nan.loc[count_nan != 0]
    nan_cols = dataset.columns[dataset.isna().any()]

    plt.bar(nan_cols, count_nans_filtered, color="green")
    plt.xlabel("Columns")
    plt.xticks(rotation=90)
    plt.ylabel("NaNs value")
    plt.show()
