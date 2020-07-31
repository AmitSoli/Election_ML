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


sfs_with_knn = [5, 31, 20, 19, 30, 26, 82, 18, 40, 4, 17, 37, 29, 39, 58, 23, 74, 85, 86, 57, 84, 78, 69, 72, 6][:15]
sfs_with_forest = [5, 31, 20, 17, 30, 26, 68, 1, 22, 82, 4, 49, 97, 37, 59, 90, 96, 18, 60, 58, 55, 85, 95, 62, 70][:8]
all_sorted_by_relief_best_first = [31, 23, 22, 29, 17, 19, 20, 5, 21, 18, 4, 30, 6, 37, 40, 36, 39, 2, 25, 26, 3, 1, 34,
                                   68, 41, 11, 54, 61, 81, 28, 75, 53, 74, 82, 55, 67, 76, 85, 86, 87, 88, 89, 90, 91,
                                   92, 56, 59, 60, 69, 84, 93, 94, 95, 96, 77, 32, 13, 57, 71, 78, 80, 58, 35, 52, 62,
                                   64, 97, 98, 99, 100, 101, 24, 65, 33, 79, 10, 63, 38, 66, 73, 83, 15, 70, 72, 0, 46,
                                   51, 9, 7, 45, 16, 14, 44, 43, 42, 12, 49, 47, 48, 27, 8, 50][:10]
sbs_with_knn = [4, 5, 17, 18, 19, 20, 21, 22, 23, 25, 26, 30, 31, 37, 82][:]
random_sfs_with_knn_1 = [39, 31, 26, 22, 20, 40, 82, 17, 18, 5, 30, 37, 4, 71, 57]
random_sfs_with_knn_2 = [5, 19, 31, 30, 82, 20, 26, 40, 17, 21, 37, 4, 39, 71, 22, 57]
correlation_coefficient_tuple = [(30, -0.4537073874262201), (20, 0.4117352057854552), (31, -0.3946257149505096),
                                 (39, 0.3687045569100953), (26, -0.34524430377495346), (29, -0.3305543190563489),
                                 (19, -0.26396072714548297), (4, -0.2180990296645353), (25, 0.2066284246216758),
                                 (3, 0.1923363806493448), (22, -0.18958494962418024), (17, -0.18873988725828733),
                                 (1, 0.18755597386787662), (37, 0.1773980807954409), (40, 0.16330407861792914),
                                 (2, 0.14652026524809678), (23, 0.10359441495866414), (35, -0.09325266129306048),
                                 (34, -0.08794649030635729), (6, 0.08221745219480696), (36, -0.08161930815643721),
                                 (41, -0.07527138731726733), (38, -0.07405480726697454), (18, -0.06218551019327321),
                                 (5, -0.03138102435041834), (55, -0.030937718180298165), (92, -0.029603543971712637),
                                 (91, -0.029603543971712637), (90, -0.029603543971712637), (89, -0.029603543971712637),
                                 (88, -0.029603543971712637), (87, -0.029603543971712637), (86, -0.029603543971712637),
                                 (85, -0.029603543971712637), (46, -0.02590120879514129), (21, -0.025422407114833383),
                                 (71, 0.023100821489028648), (50, 0.021331139865337568), (78, -0.019981859735473936),
                                 (43, 0.016839219978943755), (72, 0.016213434117916023), (16, -0.015642477105913988),
                                 (82, 0.015326436704603135), (12, 0.014322769470292691), (79, 0.013704674076727654),
                                 (0, 0.013478011550691725), (75, 0.013289109310550303), (64, -0.012963037503976415),
                                 (48, -0.012553458327220237), (101, 0.011917231694157973), (100, 0.011917231694157973),
                                 (99, 0.011917231694157973), (98, 0.011917231694157973), (97, 0.011917231694157973),
                                 (61, 0.011733119691742958), (49, 0.011660958909481668), (9, -0.011221328174037579),
                                 (24, -0.011163273152438969), (69, 0.010639082557919238), (7, -0.010404939124445586),
                                 (32, 0.010363564863782367), (84, -0.009976480264608623), (15, -0.009690410719993452),
                                 (67, -0.00940872924888145), (14, 0.00928370790590745), (56, 0.00919063170456282),
                                 (74, -0.009039597547601818), (51, -0.009027230908902386), (60, -0.008766098381051971),
                                 (28, -0.008612524765843083), (80, 0.008578196146495288), (76, 0.00830741792865875),
                                 (33, 0.00804994907520953), (77, -0.007518323778679734), (44, -0.006890411168366602),
                                 (81, -0.006656053953867886), (58, 0.006614487352623862), (53, 0.006437050696042748),
                                 (27, 0.006341474790851772), (42, -0.006320076775064494), (68, 0.006292600249319727),
                                 (52, -0.006158772807945105), (59, -0.005582933254108753), (47, 0.005515311297145206),
                                 (96, 0.00511675993891478), (95, 0.00511675993891478), (94, 0.00511675993891478),
                                 (93, 0.00511675993891478), (57, -0.005082503864158677), (13, 0.005060181849311375),
                                 (11, 0.0049386980925648725), (54, 0.0045488469471650465), (70, -0.004530127607330787),
                                 (10, -0.003957192862311269), (45, -0.0036931441224147448),
                                 (73, -0.0033022544866334147), (83, -0.0027375134221516145),
                                 (65, -0.002315339604156232), (62, 0.002085211870361079),
                                 (63, 0.0015511596535017801), (66, 0.0012954759643659431),
                                 (8, -0.00014100133590498091)][:10]


correlation_coefficient = []
for (i, corr) in correlation_coefficient_tuple:
    correlation_coefficient.append(i)

correlation_coefficient = correlation_coefficient[:5]


selection = []


data_sets = [(sfs_with_knn, 'sfs_with_knn:'),
             (sfs_with_forest, 'sfs_forest:'),
             (all_sorted_by_relief_best_first, 'relief:'),
             (sbs_with_knn, 'sbs:'),
             (random_sfs_with_knn_1, 'random_1:'),
             (random_sfs_with_knn_2, 'random_2:'),
             (correlation_coefficient, 'correlation_coefficient:')]


for (ds, ds_str) in data_sets:
    print('')
    for i in ds:
        if i not in selection:
            selection.append(i)
            print(index_to_feature_name[i])
    print(ds_str)
    print(len(selection))
    print(selection)
    print('')






print('')
print('')
print('')
print(sorted(selection))
for i in sorted(selection):
    print(index_to_feature_name[i])



























