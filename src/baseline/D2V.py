import pandas as pd
import os
import glob
from sklearn.svm import SVC
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import logging

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

def get_init_idx(y):
    idx_unknown = list(range(len(y)))
    idx_known = []
    n_rel = 0
    n_nrel = 0
    i = 0
    idx = 0
    while n_rel + n_nrel < 10:

        if y[idx] == 1 and n_rel < 5:
            val = idx_unknown.pop(i)
            idx_known.append(val)
            n_rel += 1
            idx += 1
        elif y[idx] == 0 and n_nrel < 5:
            val = idx_unknown.pop(i)
            idx_known.append(val)
            n_nrel += 1
            idx += 1
        else:
            i += 1
            idx += 1
    
    return idx_known, idx_unknown


def main(file_paths):
    print(f"Running Baseline 'D2V + SVM'\n\n")
    with open("scores_baseline.txt", "a") as f:
        f.write(f"Running Baseline 'D2V + SVM'\n\n")
    for file in file_paths:
        print(f"Evaluation for {file}:\n")
        with open("scores_baseline.txt", "a") as f:
            f.write(f"Evaluation for {file}:\n")
        rrf10 = False
        wss85 = False
        wss95 = False
        tok, y = load_data(file)
        X = create_features(tok)
        mod = SVC(probability= True)
        idx_known, idx_unknown = get_init_idx(y)
        X_known = X.iloc[idx_known,:]
        y_known = y[idx_known]
        X_unknown = X.iloc[idx_unknown,:]
        y_unknown = y[idx_unknown]

        while len(y_unknown) > 0 and not wss95:
            if len(y_known) >= 0.1*len(y) and not rrf10:

                rrf_score = sum(y_known)/sum(y)
                rrf10 = True

                logging.info(f"RRF @10: {rrf_score}")
                print(f"RRF@10 = {rrf_score}\n")
                #write the score to a file
                with open("scores_baseline.txt", "a") as f:
                    f.write(f"RRF@10 = {rrf_score}\n")
            
            if sum(y_known) > 0.85 * sum(y) and not wss85:
                wss85_score = 1 - (len(y_known)/len(y))
                wss85 = True

                logging.info(f"WSS @85: {wss85_score}")
                print(f"WSS@85 = {wss85_score}\n")
                with open("scores_baseline.txt", "a") as f:
                    f.write(f"WSS@85 = {wss85_score}\n")

            if sum(y_known) > 0.95 * sum(y) and not wss95:
                wss95_score = 1 - (len(y_known)/len(y))
                wss95 = True

                logging.info(f"WSS @95: {wss95_score}")
                print((f"WSS@95 = {wss95_score}\n"))
                with open("scores_baseline.txt", "a") as f:
                    f.write(f"WSS@95 = {wss95_score}\n\n")
            
            if rrf10 == wss85 == wss95 == True:
                break

            mod.fit(X_known, y_known)
            probability = pd.DataFrame(mod.predict_proba(X_unknown)).iloc[:,1]
            max_idx = probability.argmax()
            new_idx = idx_unknown.pop(max_idx)
            idx_known.append(new_idx)
            X_known = X.iloc[idx_known,:]
            y_known = y[idx_known]
            X_unknown = X.iloc[idx_unknown,:]
            y_unknown = y[idx_unknown]


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))
    
    datasetPath = os.path.join(cwd,"./../../data/processed_datasets")
    os.chdir(datasetPath)
    file_paths = glob.glob('*.{}'.format("csv"))
    #os.chdir(cwd)

    main(file_paths)
