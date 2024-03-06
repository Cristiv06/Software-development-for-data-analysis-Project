import numpy as np

def standardize(data):
    # numeric columns indentifying
    numeric_columns = np.all(np.isreal(data), axis=0)

    # numeric column standardize
    numeric_data = data[:, numeric_columns]

    # missing values
    numeric_data = np.nan_to_num(numeric_data)

    # convert to float
    numeric_data = numeric_data.astype(float)

    # standardize
    mean_vals = np.mean(numeric_data, axis=0)
    std_devs = np.std(numeric_data, axis=0)

    # div by 0
    std_devs[std_devs == 0] = 1.0

    standardized_data = np.zeros_like(data, dtype=float)
    standardized_data[:, numeric_columns] = (numeric_data - mean_vals) / std_devs

    return standardized_data
