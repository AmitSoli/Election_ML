import pandas as pd
import numpy as np
import sys
import math
from sklearn.model_selection import train_test_split

########################################################################################################################
# Path, input and output
########################################################################################################################
base_path = 'D:\\Courses\\ML\\Homeworks\\ML_5\\'
input_file_path = base_path + 'ElectionsData.csv'
input_file_path_to_predict = base_path + 'ElectionsData_Pred_Features.csv'
output_folder = base_path + 'prepossessing\\'

csv = pd.read_csv(input_file_path)
csv_to_predict = pd.read_csv(input_file_path)

right_feature_set_and_vote = ['Vote', 'Number_of_valued_Kneset_members', 'Yearly_IncomeK', 'Overall_happiness_score',
                              'Avg_Satisfaction_with_previous_vote', 'Most_Important_Issue', 'Will_vote_only_large_party',
                              'Garden_sqr_meter_per_person_in_residancy_area', 'Weighted_education_rank']

optimal_df = csv[right_feature_set_and_vote]

all_features = list(optimal_df)
all_features.remove('Vote')

optimal_df_to_predict = csv_to_predict[all_features]

csv_mat = optimal_df.as_matrix()
csv_mat_to_predict = optimal_df_to_predict.as_matrix()
########################################################################################################################
# Create data sets, x is the samples, y is the labels:
#   - train: precessing and save statistical parameters.
#   - valid, test: precessing with train parameters.
########################################################################################################################
x, x_test, y, y_test = train_test_split(csv_mat[:, 1:], csv_mat[:, 0], stratify=csv_mat[:, 0], test_size=0.15)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, stratify=y, test_size=0.176470588)

y_to_train = csv_mat[:, 0]

np.save(output_folder + 'y_train', y_train)
np.save(output_folder + 'y_valid', y_valid)
np.save(output_folder + 'y_test', y_test)
np.save(output_folder + 'y_to_train', y_to_train)

########################################################################################################################
#  Precessing
########################################################################################################################
train = pd.DataFrame(x_train)
valid = pd.DataFrame(x_valid)
test = pd.DataFrame(x_test)
to_predict = pd.DataFrame(csv_mat_to_predict)
to_train = pd.DataFrame(csv_mat[:, 1:])

train.columns = all_features
valid.columns = all_features
test.columns = all_features
to_predict.columns = all_features
to_train.columns = all_features
####################################################################################################################
# Outliers removal and create null features
####################################################################################################################
for f in all_features:
    train[f + 'Null'] = pd.isnull(train[f]).astype(int)
    valid[f + 'Null'] = pd.isnull(valid[f]).astype(int)
    test[f + 'Null'] = pd.isnull(test[f]).astype(int)
    to_predict[f + 'Null'] = pd.isnull(to_predict[f]).astype(int)
    to_train[f + 'Null'] = pd.isnull(to_train[f]).astype(int)


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

    dummies = pd.get_dummies(to_predict[f])
    dummies.loc[to_predict[f].isnull(), pd.unique(to_predict[f].dropna())] = np.nan
    to_predict = pd.concat([to_predict, dummies], axis=1)
    del to_predict[f]

    dummies = pd.get_dummies(to_train[f])
    dummies.loc[to_train[f].isnull(), pd.unique(to_train[f].dropna())] = np.nan
    to_train = pd.concat([to_train, dummies], axis=1)
    del to_train[f]

all_features = list(to_predict)

for data_set in [train, valid, test, to_predict, to_train]:
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
    to_predict[f] = (to_predict[f] - f_mean) / f_std
    to_train[f] = (to_train[f] - f_mean) / f_std


for f in all_features:
    if f not in normal_dst:
        f_min = train[f].min()
        f_max = train[f].max()
        train[f] = (train[f] - f_min) * (1 - (-1)) / (f_max - f_min) + (-1)
        valid[f] = (valid[f] - f_min) * (1 - (-1)) / (f_max - f_min) + (-1)
        test[f] = (test[f] - f_min) * (1 - (-1)) / (f_max - f_min) + (-1)
        to_predict[f] = (to_predict[f] - f_min) * (1 - (-1)) / (f_max - f_min) + (-1)
        to_train[f] = (to_train[f] - f_min) * (1 - (-1)) / (f_max - f_min) + (-1)

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

for f in range(x.shape[1]):
    print(f)
    for i_row in range(x.shape[0]):
        if math.isnan(x[i_row][f]):
            min_dist = sys.maxsize
            min_val = None
            for j_row in range(x.shape[0]):
                if i_row != j_row and not math.isnan(x[j_row][f]):
                    cur_dist = 0
                    if y_to_train[i_row] != y_to_train[j_row]:
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

np.save(output_folder + 'x_to_train', x)


for f in all_features:
    train_f_median = train[f].median()
    valid.loc[valid[f].isnull(), f] = train_f_median
    test.loc[test[f].isnull(), f] = train_f_median
    to_predict.loc[to_predict[f].isnull(), f] = train_f_median

x_valid = valid.as_matrix()
x_test = test.as_matrix()
x_to_predict = to_predict.as_matrix()

np.save(output_folder + 'x_valid', x_valid)
np.save(output_folder + 'x_test', x_test)
np.save(output_folder + 'x_to_predict', x_to_predict)
