# Load the sales dataset using Pandas:
import pandas as pd

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 
'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 
'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 
'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 
'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)

# Create new features by performing following transformation on inputs:
from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

# Using the entire house dataset, learn regression weights using an L1 penalty 
# of 5e2. Make sure to add "normalize=True" when creating the Lasso object. 
# Refer to the following code snippet:
from sklearn import linear_model  # using scikit-learn
all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']
model_all = linear_model.Lasso(alpha=5e2, normalize=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights
print model_all.coef_
