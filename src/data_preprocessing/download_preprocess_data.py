# Placeholder for data download and preprocessing script
import subprocess as sp
import os
import pandas as pd
import shutil
import logging


# Set up the logger globally
logger = logging.getLogger("PreProcessLogger")
logger.setLevel(logging.DEBUG)

# Create a handler and formatter
console_handler = logging.StreamHandler()


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)


cwd = os.path.dirname(os.path.realpath(__file__))
print(cwd)
os.chdir(cwd)

datasetPath = "./../../data/datasets"

def clone_dataset_repo():
    repoUrl = "https://github.com/asreview/synergy-dataset.git"
    #skipping Nagtegaal since there is no ids in the csv
    datasetFilesToCopy = [r"datasets\Cohen_2006\Cohen_2006_CalciumChannelBlockers_ids.csv",r"datasets\Kwok_2020\Kwok_2020_ids.csv",r"datasets\Bannach-Brown_2019\Bannach-Brown_2019_ids.csv"]#,r"datasets\Nagtegaal_2019\Nagtegaal_2019_ids.csv"]
    repodir = "./../../data/synergy-dataset"
    
    
    #if datasets directory already exists return and write to ouput that the datasets already exist
    
    if os.path.exists(datasetPath):
        logger.info("Datasets IDs files already exists not downloading again")
        logger.info("If you want to download again delete the datasets directory")
        return
    
    #do a no clone checkout of the repoUrl in the data directory
    #use the current cwd as the base directory
    sp.run(["git", "clone", repoUrl,repodir])
    
    #make director for datasetPath
    os.makedirs(datasetPath, exist_ok=True)
    
    for file in datasetFilesToCopy:
        # copy those files from the repo to the datasetPath
        shutil.copyfile(os.path.join(repodir, file), os.path.join(datasetPath, os.path.basename(file)))
    
    #delte the repo using git commands
    while os.path.exists(repodir):
        shutil.rmtree(repodir,ignore_errors=True)
        sp.run(["git", "clean", "-fdx", repodir])
    
    
    
    
def download_data():
    for file in os.listdir(datasetPath):
        filename = file 
        df = pd.read_csv(os.path.join(datasetPath,filename))
        outputDF = pd.DataFrame(columns=["title","abstract","text","label"])

        for row,paper in df.iterrows():
            #check if doi is not NaN
            success = False
            outputrow = None
            if pd.isna(paper.get("label_included")):
                logger.debug(f"No label found for the {row=} in {filename=} ")
                continue
                
            
            if pd.notna(paper.get("doi")) and not success:
                outputrow,success = preprocess_data(paper.get("doi"),"doi",paper.get("label_included"))
            
            if pd.notna(paper.get("pmid")) and not success:
                outputrow,success = preprocess_data(paper.get("pmid"),"pmid",paper.get("label_included"))
            
            if pd.notna(paper.get("openalex_id")) and not success:
                outputrow,success = preprocess_data(paper.get("openalex_id"),"openalex_id",paper.get("label_included"))
            else:
                
                logger.debug(f"No id found for the {row=} in {filename=} ")
            
            if success:
                outputDF = outputDF.append(outputrow)
            else:
                logger.debug(f"Could not process {row=} in {filename=} ")
            
            
        

def preprocess_data(key,method,label):
    """
    Processes one paper get the abstract,title and text

    Parameters
    ----------
    key: either a doi,pmid or openalex_id
    method: the key type

    Returns
    -------
    return row,success
    row: a pandas df row with the abstract,title,text and label
    success: a boolean value if the paper was successfully processed
    
    """
    
    # Code to preprocess data
    # eg. extract abstracts,title and text from a paper
    return None,False
    # todo download the papers and store the abstract,title,label and text in a file
        # @Alana and @Gergo

if __name__ == "__main__":
    
    
    #change this to display only info messages andso on
    console_handler.setLevel(logging.DEBUG)
    clone_dataset_repo()
    download_data()
    # preprocess_data()
    pass
