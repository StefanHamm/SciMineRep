import pandas as pd
import os
import glob
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

def load_data(file_path):
    data = pd.read_csv(file_path)
    if not {'title', 'abstract', 'label'}.issubset(data.columns):
        raise ValueError("The input CSV must contain 'title', 'abstract', and 'label' columns.")
    data['text'] = data['title'] + " " + data['abstract']
    data["tokens"] = data["text"].apply(simple_preprocess)
    return data['tokens'], data['label']


def create_features(list):
    docs = []
    for i, words in enumerate(list):
        docs.append(TaggedDocument(words= words, tags= [i]))

    mod = Doc2Vec()
    mod.build_vocab(docs)
    mod.train(docs, total_examples=mod.corpus_count, epochs=mod.epochs)

    features = [mod.dv[idx] for idx in range(len(docs))]

    return pd.DataFrame(features)

def fit_model(X, y):
    mod = SVC()
    mod.fit(X, y)
    return mod

def make_prediction(X, y, mod):
    y_pred = mod.predict(X)
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("F1:", f1_score(y, y_pred))

def main(file_paths):
    trainingX=pd.Series()
    trainingY=pd.Series()
    tests=[]
    for file in file_paths:
        tok, y = load_data(file)
        X = create_features(tok)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        trainingX = pd.concat([trainingX,X_train], ignore_index=True)
        trainingY = pd.concat([trainingY,y_train], ignore_index=True)
        tests.append([X_test,y_test])

    mod = fit_model(trainingX, trainingY)
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