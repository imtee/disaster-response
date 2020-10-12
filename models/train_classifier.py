import sys

#import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
import pickle

def load_data(database_filepath):
    '''
        loads the dataset from the database
        Parameters: the file path of the database
        Returns: the messages, categories, and list of category names   
    '''  
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('table_name', engine)
    categories = ['related', 'request', 'offer', 'aid_related', 'medical_help', 
                  'medical_products', 'search_and_rescue', 'security', 'military', 
                  'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 
                  'missing_people', 'refugees', 'death', 'other_aid','infrastructure_related', 
                  'transport', 'buildings', 'electricity','tools', 'hospitals', 'shops', 
                  'aid_centers', 'other_infrastructure','weather_related', 'floods', 'storm', 
                  'fire', 'earthquake', 'cold','other_weather', 'direct_report']
    X = df['message'].values
    y = df[categories].values
    return X, y, categories


def tokenize(text):
    '''
     process the text data: lemmatizing, case normalizing, and removing white spaces
         Parameters: text 
         Returns: clean tokens
    '''
    #tokenize text
    tokens= word_tokenize(text)
    
    #initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #iterate through each token
    clean_tokens = []
    for tok in tokens:
        #lemmatize, normalize case, and removing leading/trainling white space
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    '''
        builds the machine learning model using pipeline and GridSearch
        Parameters: none
        Returns: the model
    '''
    # This machine pipeline takes in the message column as input and output classification results
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), 
                     ('tfidf', TfidfTransformer()),
                     ('multi_clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
    'vect__max_features': (None, 5000, 10000),
        }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
        Shows the accuracy, precision, and recall of the tuned model.
        Parameters: The model, the test data input, test data output, and the category names
        Returns: none
    '''
    Y_pred = model.predict(X_test)
    for i, cat in enumerate(category_names):
     print(cat, classification_report(pd.DataFrame(Y_test)[i], pd.DataFrame(Y_pred)[i]))


def save_model(model, model_filepath):
    '''
        saves the model in a pickle file
        Parameters: the tuned model and its filepath 
        Returns: none
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()