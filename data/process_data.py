import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads datasets and merges them
        Parameters: File paths of messages and categories datasets 
        Returns: a dataframe merged from both datasets
    '''
    
    # load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets using common id
    df = messages.merge(categories, on='id')
    return df
    

def clean_data(df):
    '''
    Split categories into separate category columns and concatenates it with the original dataframe
        Parameters: the dataframe to be cleaned
        Returns: a clean dataframe 
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # extract a list of new column names for categories     
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[0:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
       categories[column] = categories[column].astype(str).str[-1]
    # convert column from string to numeric
       categories[column] = categories[column].astype(int)
    #convert related vategory to binary values, as it has "2" values
    categories['related'].unique()
    categories['related'] = categories['related'].astype(str).str.replace('2', '1')
    categories['related'] = categories['related'].astype(int)
    # Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates from dataframe
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    '''
    Saves the clean dataframe into an sqlite database
        Parameters: dataframe to be saved
                    Name of the database file
        
        Returns: none
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('table_name', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()