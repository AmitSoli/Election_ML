import numpy as np
import pandas as pd


########################################################################################################################
# Path, input and output
########################################################################################################################
base_path = 'C:\\Users\\Aviel\\PycharmProjects\\ML_2\\'
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

features = ['Occupation_Satisfaction', 'Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses',
            'Avg_Satisfaction_with_previous_vote', 'Looking_at_poles_results',
            'Garden_sqr_meter_per_person_in_residancy_area', 'Married', 'Gender', 'Voting_Time',
            'Financial_balance_score_(0-1)', '%Of_Household_Income', 'Avg_government_satisfaction',
            'Avg_education_importance', 'Avg_environmental_importance', 'Avg_Residancy_Altitude', 'Yearly_ExpensesK',
            '%Time_invested_in_work', 'Yearly_IncomeK', 'Avg_monthly_expense_on_pets_or_plants',
            'Avg_monthly_household_cost', 'Will_vote_only_large_party', 'Phone_minutes_10_years',
            'Avg_size_per_room', 'Weighted_education_rank', '%_satisfaction_financial_policy',
            'Avg_monthly_income_all_years', 'Last_school_grades', 'Age_group', 'Number_of_differnt_parties_voted_for',
            'Political_interest_Total_Score', 'Number_of_valued_Kneset_members', 'Overall_happiness_score',
            'Num_of_kids_born_last_10_years', 'Financial_agenda_matters', 'Education', 'Environment', 'Financial',
            'Foreign_Affairs', 'Healthcare', 'Military', 'Other', 'Social', 'Car', 'Foot_or_bicycle',
            'Motorcycle_or_truck', 'Public_or_other', 'Hightech', 'Industry_or_other', 'Public_Sector',
            'Services_or_Retail', 'Student_or_Unemployed', 'Occupation_SatisfactionNull',
            'Avg_monthly_expense_when_under_age_21Null', 'AVG_lottary_expansesNull',
            'Avg_Satisfaction_with_previous_voteNull', 'Looking_at_poles_resultsNull',
            'Garden_sqr_meter_per_person_in_residancy_areaNull', 'MarriedNull', 'GenderNull', 'Voting_TimeNull',
            'Financial_balance_score_(0-1)Null', '%Of_Household_IncomeNull', 'Avg_government_satisfactionNull',
            'Avg_education_importanceNull', 'Avg_environmental_importanceNull', 'Avg_Residancy_AltitudeNull',
            'Yearly_ExpensesKNull', '%Time_invested_in_workNull', 'Yearly_IncomeKNull',
            'Avg_monthly_expense_on_pets_or_plantsNull', 'Avg_monthly_household_costNull',
            'Will_vote_only_large_partyNull', 'Phone_minutes_10_yearsNull', 'Avg_size_per_roomNull',
            'Weighted_education_rankNull', '%_satisfaction_financial_policyNull', 'Avg_monthly_income_all_yearsNull',
            'Last_school_gradesNull', 'Age_groupNull', 'Number_of_differnt_parties_voted_forNull',
            'Political_interest_Total_ScoreNull', 'Number_of_valued_Kneset_membersNull',
            'Overall_happiness_scoreNull', 'Num_of_kids_born_last_10_yearsNull', 'Financial_agenda_mattersNull',
            'EducationNull', 'EnvironmentNull', 'FinancialNull', 'Foreign_AffairsNull', 'HealthcareNull',
            'MilitaryNull', 'OtherNull', 'SocialNull', 'CarNull', 'Foot_or_bicycleNull', 'Motorcycle_or_truckNull',
            'Public_or_otherNull', 'HightechNull', 'Industry_or_otherNull', 'Public_SectorNull',
            'Services_or_RetailNull', 'Student_or_UnemployedNull']


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

