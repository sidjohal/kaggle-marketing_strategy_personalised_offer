# Essential libraries
import pandas as pd, numpy as np, matplotlib.pyplot as pyplot


# Data loading
raw = pd.read_csv("data/train_data.csv")
features = raw.drop("Offer Accepted", axis=1)
label = raw["Offer Accepted"]


# Data statistics
#raw.shape, raw.shape, raw.isna().sum(), raw.info()


'''
To handle NA values we can either impute or drop the column
To impute, we use SimpleImputer or KNNImputer
'''

# all the data preprocessing to be done on train set needs to be done on test set
# so we'll define a pipeline that does that


