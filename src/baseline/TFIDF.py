import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import os
import glob

def load_data(file_path):
    data = pd.read_csv(file_path)
    if not {'title', 'abstract', 'label'}.issubset(data.columns):
        raise ValueError("The input CSV must contain 'title', 'abstract', and 'label' columns.")
    data['text'] = data['title'] + " " + data['abstract']
    return data['text'], data['label']


def build_and_train_model(X, y):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', MultinomialNB(fit_prior=False))
    ])
    pipeline.fit(X, y)
    return pipeline

def make_prediction(X,y,pipeline):
    y_pred = pipeline.predict(X)
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("F1:", f1_score(y, y_pred))


def main(file_paths):
    trainingX=pd.Series()
    trainingY=pd.Series()
    tests=[]
    for file in file_paths:
        X, y = load_data(file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        trainingX = pd.concat([trainingX,X_train], ignore_index=True)
        trainingY = pd.concat([trainingY,y_train], ignore_index=True)
        tests.append([X_test,y_test])
    
    model = build_and_train_model(trainingX, trainingY)
    for i in range(len(tests)):
        print(f'\n\nPrediction for {file_paths[i]}:\n')
        make_prediction(tests[i][0],tests[i][1],model)
    
    all_X = pd.concat([item[0] for item in tests], ignore_index=True)
    all_Y = pd.concat([item[1] for item in tests], ignore_index=True)
    print('\n\nOverall prediction:\n')
    make_prediction(all_X,all_Y,model)

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    datasetPath = os.path.join(cwd,"./../../data/processed_datasets")
    os.chdir(datasetPath)
    file_paths = glob.glob('*.{}'.format("csv"))

    main(file_paths)