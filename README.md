# Disaster Response Pipeline Project
### Table of Contents

1. [Installation](#installation)
2. [Project Description](#description)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
It is recommended to run the project in a virtual environment after cloning it. Please install the requirements using "pip3 install -r requirements.txt". Make sure you use Python versions 3.*

## Project Description<a name="description"></a>

This project is whithin [Udacity](https://www.udacity.com/) datascience nanodegree. It aims to analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages.

This project is formed by three parts:
 - ETL pipeline: an Extract, Transform, and Load process to read the dataset, clean the data, and then store it in a SQLite database.
 
 - Machine Learning Pipeline: a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification).
 
 - Flask App: to display the results.
 
 ## File Descriptions <a name="files"></a>
 
- ../data/run.py is the executable for the Flask web app
- requirements.txt: contains the environment requirements to run the program
- app folder contains:
     - templates: a folder containing:
       - index.html: it renders the homepage
       - go.html: it renders the results of the message classifier
     - run.py: defines the app routes
- data folder contains:
     - disaster_categories.csv: the disaster categories csv file
     - disaster_messages.csv: the disaster messages csv file
     - DisasterResponse.db: the clean and merged database from disaster categories and messages.
     - process_data.py: the script of ETL process to clean and store the data in database
- models folder contains:
     - classifier.pkl: the RandomForestClassifier pickle file
     - train_classifier.py: the script to train the machine learning model

 ## Results <a name="results"></a>
 To diplay the results, follow these instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. In the terminal, use this command to get the link for viewing the app:
    env | grep WORK
    The link will be: http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN replacing WORKSPACEID and WORKSPACEDOMAIN with your values.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [Figure Eight](https://appen.com/) and [Udacity](https://www.udacity.com/) for the course. This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).



