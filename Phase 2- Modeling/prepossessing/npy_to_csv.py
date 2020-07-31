import numpy as np
import pandas as pd


########################################################################################################################
# Path, input and output
########################################################################################################################
base_path = 'D:\\Courses\\ML\\Homeworks\\ML_3\\'
input_file_path = base_path + 'prepossessing\\'
output_path = base_path + 'csv_dada_sets\\'

x_train = np.load(input_file_path + 'x_train.npy')
y_train = np.load(input_file_path + 'y_train.npy')
x_valid = np.load(input_file_path + 'x_valid.npy')
y_valid = np.load(input_file_path + 'y_valid.npy')
x_test = np.load(input_file_path + 'x_test.npy')
y_test = np.load(input_file_path + 'y_test.npy')

x_train_org = np.load(input_file_path + 'x_train_org.npy')
x_valid_org = np.load(input_file_path + 'x_valid_org.npy')
x_test_org = np.load(input_file_path + 'x_test_org.npy')

features = ['Number_of_valued_Kneset_members', 'Yearly_IncomeK', 'Overall_happiness_score',
            'Avg_Satisfaction_with_previous_vote', 'Will_vote_only_large_party',
            'Garden_sqr_meter_per_person_in_residancy_area', 'Weighted_education_rank',
            'Number_of_valued_Kneset_membersNull', 'Yearly_IncomeKNull', 'Overall_happiness_scoreNull',
            'Avg_Satisfaction_with_previous_voteNull', 'Most_Important_IssueNull', 'Will_vote_only_large_partyNull',
            'Garden_sqr_meter_per_person_in_residancy_areaNull', 'Weighted_education_rankNull', 'Education',
            'Environment', 'Financial', 'Foreign_Affairs', 'Healthcare', 'Military', 'Other', 'Social']


index_to_feature_name = {}
for i in range(len(features)):
    index_to_feature_name[i] = features[i]


data_sets = [(x_train, y_train, 'train_mod_df.csv'),
             (x_valid, y_valid, 'valid_mod_df.csv'),
             (x_test, y_test, 'test_mod_df.csv'),
             (x_train_org, y_train, 'train_org_df.csv'),
             (x_valid_org, y_valid, 'valid_org_df.csv'),
             (x_test_org, y_test, 'test_org_df.csv')]

for (x, y, file_name) in data_sets:
    df = pd.DataFrame(x)
    df['Vote'] = y
    cols = df.columns.tolist()
    df = df[[cols[-1]] + cols[:-1]]
    df.rename(columns=index_to_feature_name, inplace=True)
    df.to_csv(output_path + file_name, sep=',', index=0)

