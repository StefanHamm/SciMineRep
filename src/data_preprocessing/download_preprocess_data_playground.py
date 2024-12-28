import subprocess as sp
import os
import pandas as pd
import shutil
import logging
import requests

# Set up the logger globally
logger = logging.getLogger("PreProcessLogger")
logger.setLevel(logging.DEBUG)

# Create a handler and formatter
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# Set up dataset path
#cwd = os.getcwd()
cwd = os.path.dirname(os.path.realpath(__file__))
print(cwd)
os.chdir(cwd)
datasetPath = "./../../data/datasets"

# Fetches metadata from PubMed using PMID
def get_pubmed_metadata(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=xml"
    response = requests.get(url)
    if response.status_code == 200:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.text)
        doc = root.find(".//DocSum")
        title = doc.find(".//Item[@Name='Title']").text if doc is not None else "No title available"
        abstract = doc.find(".//Item[@Name='Source']").text if doc is not None else "No abstract available"
        return title, abstract
    else:
        return None, None

# Fetches metadata from CrossRef using DOI
def get_crossref_metadata(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        title_list = data["message"].get("title", ["No title available"])
        title = title_list[0] if title_list else "No title available"
        abstract = data["message"].get("abstract", "No abstract available")
        return title, abstract
    else:
        return None, None

# Fetches metadata from OpenAlex using OpenAlex ID
def get_openalex_metadata(openalex_id):
    url = f"https://api.openalex.org/works/{openalex_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        title = data["title"] if "title" in data else "No title available"
        abstract = data.get("abstract", "No abstract available")
        return title, abstract
    else:
        return None, None

# Preprocess data for a given row based on available identifiers (PMID, DOI, OpenAlex)
def preprocess_data(paper):
    title, abstract = None, None
    
    # Try DOI
    if pd.notna(paper.get("doi")):
        title, abstract = get_crossref_metadata(paper["doi"][16:])
    
    # If DOI doesn't exist or no data was found, try PMID
    if ((title is None) or (title=='No title available')) and pd.notna(paper.get("pmid")):
        title, abstract = get_pubmed_metadata(paper["pmid"][-8:])

    # If DOI also doesn't exist or no data was found, try OpenAlexID
    if ((title is None) or (title=='No title available')) and pd.notna(paper.get("openalex_id")):
        title, abstract = get_openalex_metadata(paper["openalex_id"][21:])
    
    # Return a dictionary if metadata was found
    if title and abstract:
        return {"title": title, "abstract": abstract, "label": paper["label_included"]}
    else:
        return None

# Function to clone the dataset repository
def clone_dataset_repo():
    repoUrl = "https://github.com/asreview/synergy-dataset.git"
    datasetFilesToCopy = [
        r"datasets/Cohen_2006/Cohen_2006_CalciumChannelBlockers_ids.csv",
        r"datasets/Kwok_2020/Kwok_2020_ids.csv",
        r"datasets/Bannach-Brown_2019/Bannach-Brown_2019_ids.csv"
    ]
    repodir = "./../../data/synergy-dataset"
    
    if os.path.exists(datasetPath):
        logger.info("Datasets IDs files already exist. Not downloading again.")
        return
    
    # Clone the repository
    sp.run(["git", "clone", repoUrl, repodir])
    
    # Create the dataset directory and copy the files
    os.makedirs(datasetPath, exist_ok=True)
    for file in datasetFilesToCopy:
        shutil.copyfile(os.path.join(repodir, file), os.path.join(datasetPath, os.path.basename(file)))
    
    # Clean up the cloned repository
    shutil.rmtree(repodir, ignore_errors=True)

# Function to download and process the dataset
def download_data():
    for file in os.listdir(datasetPath):
        filename = file
        try:
            # Attempt to read the CSV file with a specific encoding
            df = pd.read_csv(os.path.join(datasetPath, filename), encoding='ISO-8859-1')  # Or try 'latin1', 'utf-16'
        except UnicodeDecodeError as e:
            logger.error(f"Error reading {filename}: {e}")
            continue  # Skip this file and continue to the next one
        
        outputDF = pd.DataFrame(columns=["title", "abstract", "label"])

        for row, paper in df.iterrows():
            success = False
            outputrow = None
            if pd.isna(paper.get("label_included")):
                logger.debug(f"No label found for row {row} in {filename}")
                continue
            
            if pd.notna(paper.get("doi")) and not success:
                outputrow = preprocess_data(paper)
                if outputrow:
                    success = True
            
            if pd.notna(paper.get("pmid")) and not success:
                outputrow = preprocess_data(paper)
                if outputrow:
                    success = True
            
            if pd.notna(paper.get("openalex_id")) and not success:
                outputrow = preprocess_data(paper)
                if outputrow:
                    success = True
            
            if success and outputrow:
                outputDF = pd.concat([outputDF, pd.DataFrame([outputrow])], ignore_index=True)
            else:
                logger.debug(f"Could not process row {row} in {filename}")
        
        # Save the resulting data to CSV and Pickle files
        outputDF.to_csv(f"{filename}_processed.csv", index=False)
        outputDF.to_pickle(f"{filename}_processed.pkl")


if __name__ == "__main__":
    # Change this to display only info messages and so on
    console_handler.setLevel(logging.INFO)
    clone_dataset_repo()
    download_data()
