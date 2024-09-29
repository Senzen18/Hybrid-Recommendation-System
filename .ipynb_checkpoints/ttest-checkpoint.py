from scipy.stats import ttest_ind
import pandas as pd
def ttest():
    Recommender_Response1 = pd.read_csv('Recommender_Responsev1.csv',usecols=['Users','Recommender_Ratings'])  
    Recommender_Response2 = pd.read_csv('Recommender_Responsev2.csv',usecols=['Users','Recommender_Ratings'])
    evaluation = ''
    
    if ttest_ind(Recommender_Response1['Recommender_Ratings'].values,Recommender_Response2['Recommender_Ratings'].values).pvalue < 0.05:
        evaluation += "Reject Null Hypothesis μ1 = μ2  "
        if ttest_ind(Recommender_Response1['Recommender_Ratings'].values,Recommender_Response2['Recommender_Ratings'].values,alternative = 'less').pvalue < 0.05:
            evaluation += 'and μ1<μ12'
        else:
            evaluation +='and μ1>μ12' 
    else:
        evaluation += "Null Hypothesis μ1 = μ2 is True"
    return evaluation
        