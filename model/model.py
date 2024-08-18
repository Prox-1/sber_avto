import re
import dill
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC



def create_target_action(df):
    def create_target_column(value):
        if value in ['sub_car_claim_click', 'sub_car_claim_submit_click',
'sub_open_dialog_click', 'sub_custom_question_submit_click',
'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
'sub_car_request_submit_click']:
            return 1
        else:
            return 0
    df['target_action'] = df['event_action'].apply(create_target_column)
    df = df.drop('event_action', axis = 1)
    return df

def filter_data(df):
    df_copy = df
    columns_to_drop = ['session_id', 'client_id','utm_keyword', 'visit_date','visit_time','visit_number','geo_city']
    return df_copy.drop(columns_to_drop, axis=1)

def not_na(df):

    def replace_strings_with_digits(strings):
        return str("(not set)" if re.search(r'\d', strings) else strings )
    
    df['device_os'] = df['device_os'].apply(lambda x: '(not set)' if pd.isna(x) else x)
    df['device_brand'] = df['device_brand'].apply(lambda x: '(not set)' if pd.isna(x) else x)
    for i in ['utm_source', 'utm_campaign', 'utm_adcontent', 'device_model', 'geo_country']:
        df[i] = df[i].apply(lambda x: str(x).lower() )
    for i in ['utm_source', 'utm_campaign', 'utm_adcontent','device_model']:
        moda_value = df[i].value_counts().idxmax()
        df[i] = df[i].apply(lambda x: moda_value if pd.isna(x) else x)
    df['device_screen_resolution'] = df['device_screen_resolution'].apply(lambda x: 0.0 if x == '(not set)' else float(x.replace('x','.' )))
    df['geo_country'] = df['geo_country'].apply(lambda x: replace_strings_with_digits(x))
    df = df.drop(['session_id', 'client_id','utm_keyword', 'visit_date','visit_time','visit_number','geo_city'], axis = 1, errors='ignore')
    return df.drop_duplicates()


def main():
    df_1 = pd.read_csv("model\ga_sessions.csv")
    df_2 = pd.read_csv("model\ga_hits.csv")
    df = (pd.merge(df_1, df_2[['session_id', 'event_action']], on='session_id', how='left')).drop_duplicates()
    df = not_na(filter_data(create_target_action(df)))
    X = df.drop('target_action', axis=1)
    y = df['target_action']
    
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor_2 = ColumnTransformer(transformers=[
    ('numerical', numerical_transformer, numerical_features),
    ('categorical', categorical_transformer, categorical_features)
    ])

    model = RandomForestClassifier(random_state=42)

    best_score = .0
    best_pipe= None
    pipe = Pipeline(steps=[
    ('preprocessor_2', preprocessor_2),
    ('classifier', model)
    ])
    score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
    print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
    best_score = score.mean()
    best_pipe = pipe
    
    best_pipe.fit(X, y)
    with open('target_action.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'target action predicrion model',
                'author': 'Kurganov Daniil (prox)',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)
    print('end')
    

if __name__ == '__main__':
    main()