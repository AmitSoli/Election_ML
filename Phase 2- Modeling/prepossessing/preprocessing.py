import pandas as pd
import numpy as np
import sys
import math
from sklearn.model_selection import train_test_split

########################################################################################################################
# Path, input and output
########################################################################################################################
base_path = 'D:\\Courses\\ML\\Homeworks\\ML_3\\'
input_file_path = base_path + 'ElectionsData.csv'
output_folder = base_path + 'prepossessing\\'

csv = pd.read_csv(input_file_path)

right_feature_set_and_vote = ['Vote', 'Number_of_valued_Kneset_members', 'Yearly_IncomeK', 'Overall_happiness_score',
                              'Avg_Satisfaction_with_previous_vote', 'Most_Important_Issue', 'Will_vote_only_large_party',
                              'Garden_sqr_meter_per_person_in_residancy_area', 'Weighted_education_rank']

optimal_df = csv[right_feature_set_and_vote]
csv_mat = optimal_df.as_matrix()

########################################################################################################################
# Create data sets, x is the samples, y is the labels:
#   - train: precessing and save statistical parameters.
#   - valid, test: precessing with train parameters.
########################################################################################################################
x, x_test, y, y_test = train_test_split(csv_mat[:, 1:], csv_mat[:, 0], stratify=csv_mat[:, 0], test_size=0.15)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, stratify=y, test_size=0.176470588)


np.save(output_folder + 'x_train_org', x_train)
np.save(output_folder + 'x_valid_org', x_valid)
np.save(output_folder + 'x_test_org', x_test)

np.save(output_folder + 'y_train', y_train)
np.save(output_folder + 'y_valid', y_valid)
np.save(output_folder + 'y_test', y_test)

########################################################################################################################
#  Precessing
########################################################################################################################
train = pd.DataFrame(x_train)
valid = pd.DataFrame(x_valid)
test = pd.DataFrame(x_test)

all_features = list(optimal_df)
all_features.remove('Vote')

train.columns = all_features
valid.columns = all_features
test.columns = all_features

####################################################################################################################
# Outliers removal and create null features
####################################################################################################################
for f in all_features:
    train[f + 'Null'] = pd.isnull(train[f]).astype(int)
    valid[f + 'Null'] = pd.isnull(valid[f]).astype(int)
    test[f + 'Null'] = pd.isnull(test[f]).astype(int)


####################################################################################################################
# Labels Handling :
#   - Categorical to One-Hot.
#   - Ordered to numbers.
####################################################################################################################
categorical_features = ['Most_Important_Issue']
for f in categorical_features:
    dummies = pd.get_dummies(train[f])
    dummies.loc[train[f].isnull(), pd.unique(train[f].dropna())] = np.nan
    train = pd.concat([train, dummies], axis=1)
    del train[f]

    dummies = pd.get_dummies(valid[f])
    dummies.loc[valid[f].isnull(), pd.unique(valid[f].dropna())] = np.nan
    valid = pd.concat([valid, dummies], axis=1)
    del valid[f]

    dummies = pd.get_dummies(test[f])
    dummies.loc[test[f].isnull(), pd.unique(test[f].dropna())] = np.nan
    test = pd.concat([test, dummies], axis=1)
    del test[f]

all_features = list(train)

for data_set in [train, valid, test]:
    data_set['Will_vote_only_large_party'] = data_set['Will_vote_only_large_party'].map({'No': -1, 'Maybe': 0, 'Yes': 1})

print(all_features)
########################################################################################################################
# Scaling.
########################################################################################################################
normal_dst = ['Avg_Satisfaction_with_previous_vote', 'Garden_sqr_meter_per_person_in_residancy_area', 'Yearly_IncomeK',
              'Weighted_education_rank', 'Overall_happiness_score']

for f in normal_dst:
    f_mean = train[f].mean()
    f_std = train[f].std()
    train[f] = (train[f] - f_mean) / f_std
    valid[f] = (valid[f] - f_mean) / f_std
    test[f] = (test[f] - f_mean) / f_std

for f in all_features:
    if f not in normal_dst:
        f_min = train[f].min()
        f_max = train[f].max()
        train[f] = (train[f] - f_min) * (1 - (-1)) / (f_max - f_min) + (-1)
        valid[f] = (valid[f] - f_min) * (1 - (-1)) / (f_max - f_min) + (-1)
        test[f] = (test[f] - f_min) * (1 - (-1)) / (f_max - f_min) + (-1)

########################################################################################################################
# Filling missing values:
#   - train: closest fit method.
#   - valid, test: median.
########################################################################################################################

x = train.as_matrix()
for f in range(x.shape[1]):
    print(f)
    for i_row in range(x.shape[0]):
        if math.isnan(x[i_row][f]):
            min_dist = sys.maxsize
            min_val = None
            for j_row in range(x.shape[0]):
                if i_row != j_row and not math.isnan(x[j_row][f]):
                    cur_dist = 0
                    if y_train[i_row] != y_train[j_row]:
                        cur_dist += 5
                    for k in range(x.shape[1]):
                        if math.isnan(x[j_row][k]) or math.isnan(x[i_row][k]):
                            if k != f:
                                cur_dist += 2
                        else:
                            cur_dist += abs(x[j_row][k] - x[i_row][k])
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        min_val = x[j_row][f]

            x[i_row][f] = min_val

np.save(output_folder + 'x_train', x)


for f in all_features:
    train_f_median = train[f].median()
    valid.loc[valid[f].isnull(), f] = train_f_median
    test.loc[test[f].isnull(), f] = train_f_median

x_valid = valid.as_matrix()
x_test = test.as_matrix()

np.save(output_folder + 'x_valid', x_valid)
np.save(output_folder + 'x_test', x_test)
