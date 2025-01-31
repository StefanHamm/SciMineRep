import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import os
import glob
import sys 
from dataloader import newReviewDataset


def make_prediction(X,y,mod):
    y_pred = mod.predict(X)
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("F1:", f1_score(y, y_pred))

def main(file_paths):
    trainingX=pd.Series()
    trainingY=pd.Series()
    tests=[]
    for file in file_paths:
        DS = newReviewDataset(data_path= file)
        X = DS.embeddings
        y = DS.labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        trainingX = pd.concat([trainingX,X_train], ignore_index=True)
        trainingY = pd.concat([trainingY,y_train], ignore_index=True)
        tests.append([X_test,y_test])
    
    mod = SVC()
    mod.fit(trainingX, trainingY)
    for i in range(len(tests)):
        print(f'\n\nPrediction for {file_paths[i]}:\n')
        make_prediction(tests[i][0],tests[i][1],mod)
    
    all_X = pd.concat([item[0] for item in tests], ignore_index=True)
    all_Y = pd.concat([item[1] for item in tests], ignore_index=True)
    print('\n\nOverall prediction:\n')
    make_prediction(all_X,all_Y,mod)



if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    datasetPath = os.path.join(cwd,"./../../data/processed_datasets")
    os.chdir(datasetPath)
    file_paths = glob.glob('*.{}'.format("csv"))

    main(file_paths)