from scipy.stats import chi2_contingency, norm
import pandas as pd
import numpy as np
import itertools


def process_columns(df: pd.DataFrame) -> list:
    df.columns = [col.lower() for col in df.columns]
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df.columns = [col.replace('/', '_') for col in df.columns]
    df['other_activities'] = df['other_activities'].fillna('')
    df['other_interactions'] = df['other_interactions'].fillna('')

    df['other_activities'] = df['other_activities'].apply(lambda x: str(x) + ' ')
    df['other_'] = df['other_activities'] + df['other_interactions'] # df['basic_action_text'] +

    df['primary_fur_color'] = df['primary_fur_color'].fillna('')
    df['other_activities'] = df['other_activities'].fillna('')
    df['other_interactions'] = df['other_interactions'].fillna('')
    df['color_notes'] = df['color_notes'].fillna('')
    df['specific_location'] = df['specific_location'].fillna('')
    df['highlight_fur_color'] = df['highlight_fur_color'].fillna('')
    df['location'] = df['location'].fillna('')
    df['age'] = df['age'].fillna('')

    df['combination_of_primary_and_highlight_color'] = df['combination_of_primary_and_highlight_color'].fillna('')
    df['above_ground_sighter_measurement'] = df['above_ground_sighter_measurement'].fillna(0)

    cols = [
        'primary_fur_color', 'other_activities', 'other_interactions', 'color_notes', 'specific_location', 
        'highlight_fur_color', 'combination_of_primary_and_highlight_color', 'location', 'age'
        ]
    for c in cols:
        df[c] = df[c].apply(lambda x: x.lower())

    return df

def define_target(df):
    # defining variable
    def find_approach(row):
        _approach_activities = 1 if 'approach' in row['other_activities'] else 0
        _approach_interactions = 1 if 'approach' in row['other_interactions'] else 0
        res = 1 if _approach_activities+_approach_interactions > 0 else 0
        return res

    def find_beg(row):
        _approach_activities = 1 if 'beg' in row['other_activities'] else 0
        _approach_interactions = 1 if 'beg' in row['other_interactions'] else 0
        res = 1 if _approach_activities+_approach_interactions > 0 else 0
        return res

    df['playing'] = df['other_activities'].apply(lambda x: 1 if 'play' in x else 0)
    df['other_approach'] = df.apply(find_approach, axis=1)
    df['other_beg'] = df.apply(find_beg, axis=1)

    df['friendly'] = df.apply(lambda row: 1 if (row['approaches']==1) | (row['playing']==1) | (row['other_approach']==1) | (row['other_beg']==1) else 0, axis=1)
    return df

df = pd.read_csv("topic_modeling/data/2018_Central_Park_Squirrel_Census_-_Squirrel_Data.csv", parse_dates=["Date"])
df = process_columns(df)
df = define_target(df)

contingency_table = [
    [
        len(df.loc[(df['shift'] == 'AM') & (df['age'] == 'adult')]), 
        len(df.loc[(df['shift'] == 'AM') & (df['age'] == 'juvenile')])
        ],
    [
        len(df.loc[(df['shift'] == 'PM') & (df['age'] == 'adult')]), 
        len(df.loc[(df['shift'] == 'PM') & (df['age'] == 'juvenile')])
    ]
]

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square Statistic:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:", expected)

alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis: There is a significant difference in squirrel sightings across groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in squirrel sightings across groups.")


""" multiple testing """

data = {
    'primary_fur_color': ['black', 'black', 'cinnamon', 'cinnamon', 'gray', 'gray'],
    'friendly': ['yes', 'no', 'yes', 'no', 'yes', 'no'],
    'count': [len(df.loc[(df['primary_fur_color']=='black') & (df['friendly']==1)]), 
              len(df.loc[(df['primary_fur_color']=='black') & (df['friendly']==0)]), 
              len(df.loc[(df['primary_fur_color']=='cinnamon') & (df['friendly']==1)]), 
              len(df.loc[(df['primary_fur_color']=='cinnamon') & (df['friendly']==0)]), 
              len(df.loc[(df['primary_fur_color']=='gray') & (df['friendly']==1)]), 
              len(df.loc[(df['primary_fur_color']=='gray') & (df['friendly']==0)])]
}
data = pd.DataFrame(data)

# Create a contingency table
contingency_table = pd.pivot_table(data, values='count', index='primary_fur_color', columns='friendly', aggfunc='sum').fillna(0)
print("Contingency Table:\n", contingency_table)

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:\n", expected)

# If chi-square is significant, perform pairwise z-tests
if p < 0.05:
    print("Overall significant difference found, proceeding with pairwise comparisons.")
    
    # Flatten the contingency table for pairwise comparisons
    friendly = contingency_table['yes'].values
    not_friendly = contingency_table['no'].values
    total = friendly + not_friendly
    
    # Define a function for pairwise z-test for proportions
    def pairwise_z_test(i, j):
        p1 = friendly[i] / total[i]
        p2 = friendly[j] / total[j]
        p_pool = (friendly[i] + friendly[j]) / (total[i] + total[j])
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / total[i] + 1 / total[j]))
        z = (p1 - p2) / se
        p_val = 2 * (1 - norm.cdf(abs(z)))
        return p_val
    
    # Perform pairwise comparisons and apply Bonferroni correction
    groups = range(len(total))
    pairs = list(itertools.combinations(groups, 2))
    p_values = []
    
    for (i, j) in pairs:
        p_val = pairwise_z_test(i, j)
        p_values.append((i, j, p_val))
    
    # Bonferroni correction
    bonferroni_alpha = 0.05 / len(pairs)
    
    for (i, j, p_val) in p_values:
        if p_val < bonferroni_alpha:
            print(f"Significant difference between group {i} and group {j} (p-value: {p_val:.4f}) after Bonferroni correction.")
        else:
            print(f"No significant difference between group {i} and group {j} (p-value: {p_val:.4f}) after Bonferroni correction.")
else:
    print("No significant difference found in the overall test.")