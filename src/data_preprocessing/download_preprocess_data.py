import subprocess as sp
import os
import pandas as pd
import shutil
import logging
import requests
import asyncio
import aiohttp
import json
from tqdm.asyncio import tqdm as tqdm_async
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
processedDataPath = "./../../data/processed_datasets"

# Add rate limiting
# limit is number of concurrent requests
# interval is the time in seconds between each request
# rate lmit per second = limit/interval
RATE_LIMITS = {
    "openalex": {"limit": 1, "interval": 0.2},  # 5 requests per second
    "eutils": {"limit": 1, "interval": 0.2},   # 5 requests per second
    "crossref": {"limit": 5, "interval": 0.1}, # 5 requests per second
}

# Create semaphores for each service
semaphores = {
    "openalex": asyncio.Semaphore(RATE_LIMITS["openalex"]["limit"]),
    "eutils": asyncio.Semaphore(RATE_LIMITS["eutils"]["limit"]),
    "crossref": asyncio.Semaphore(RATE_LIMITS["crossref"]["limit"]),
}
# Asynchronous fetch function
async def fetch(service,session, url,semaphores):
    semaphore = semaphores[service]
    logger.debug(f"Waiting for semaphore {url} with {service} service and semaphore {semaphore._value}")
    
    async with semaphore:
        logger.debug(f"Fetching {url} with {service} service and semaphore {semaphore._value}")
        try:
            async with session.get(url) as response:
                logger.debug(f"Got response {url} with {service} service and semaphore {semaphore._value}")
                logger.debug(f"Sleeping {url} for {RATE_LIMITS[service]['interval']/RATE_LIMITS[service]['limit']} seconds")
                await asyncio.sleep(RATE_LIMITS[service]["interval"]/RATE_LIMITS[service]["limit"])
                
                if response.status == 200:
                    return await response.text()
                elif response.status == 404:
                    logger.warning(f"Data not found for {service} service with URL: {url}")
                    return None
                elif response.status == 429:
                    logger.error(f"Rate limit exceeded for {service} service reached")
                    logger.error(f"Try again after a few seconds but pleas change the rate limit")
                    # exiting the program as rate limit exceeded
                    exit(1)
                else:
                    logger.error(f"Error {response.status} for URL: {url}")
                    return None
        except Exception as e:
            logger.error(f"Request failed in fetch(): {e}")
            return None

# Fetches metadata from PubMed using PMID
async def get_pubmed_metadata(session,pmid,semaphores):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=xml"
    text = await fetch("eutils",session, url,semaphores)
    if text:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(text)
        doc = root.find(".//DocSum")
        if doc is None:
                logger.warning(f"No data found for PMID: {pmid} with URL: {url}")
                return None, None
        title = doc.find(".//Item[@Name='Title']").text 
        abstract = doc.find(".//Item[@Name='Source']").text 
        return title, abstract
    else:
        return None, None

# Fetches metadata from CrossRef using DOI
async def get_crossref_metadata(session,doi,semaphores):
    url = f"https://api.crossref.org/works/{doi}"
    text = await fetch("crossref",session, url,semaphores)
    if text:
        data = json.loads(text)
        title_list = data["message"].get("title", [None])
        title = title_list[0] if title_list else None
        abstract = data["message"].get("abstract", None)
        return title, abstract
    else:
        return None, None

# Fetches metadata from OpenAlex using OpenAlex ID
async def get_openalex_metadata(session,openalex_id,semaphores):
    url = f"https://api.openalex.org/works/{openalex_id}"
    text = await fetch("openalex",session, url,semaphores)
    if text:
        data = json.loads(text)
        title = data["title"] if "title" in data else "No title available"
        abstract = data.get("abstract", None)
        return title, abstract
    else:
        return None, None


async def preprocess_data(session, paper, semaphores):
    title, abstract = None, None

    # Try DOI via CrossRef
    if pd.notna(paper.get("doi")):
        title_cr, abstract_cr = await get_crossref_metadata(
            session, paper["doi"], semaphores
        )
        if title_cr:
            title = title_cr
        if abstract_cr and abstract_cr != "No abstract available":
            abstract = abstract_cr


    # Try PubMed via PMID
    if pd.notna(paper.get("pmid")) and (not title or not abstract):
        title_pm, abstract_pm = await get_pubmed_metadata(
            session, paper["pmid"].split("/")[-1], semaphores
        )
        if not title and title_pm:
            title = title_pm
        if not abstract and abstract_pm and abstract_pm != "No abstract available":
            abstract = abstract_pm

    # Try OpenAlex via OpenAlex ID
    if pd.notna(paper.get("openalex_id")) and (not title or not abstract):
        title_oa_id, abstract_oa_id = await get_openalex_metadata(
            session, paper["openalex_id"].split("/")[-1], semaphores
        )
        if not title and title_oa_id:
            title = title_oa_id
        if not abstract and abstract_oa_id and abstract_oa_id != "No abstract available":
            abstract = abstract_oa_id
    
     # Return a dictionary if metadata was found
    if title and abstract:
        return {"title": title, "abstract": abstract, "label": paper["label_included"]}
    else:
        return None

# Function to clone the dataset repository
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
    



async def process_file(filename,semaphores):
    try:
        df = pd.read_csv(os.path.join(datasetPath, filename), encoding="ISO-8859-1")
    except UnicodeDecodeError as e:
        logger.error(f"Error reading {filename}: {e}")
        return

    outputDF = pd.DataFrame(columns=["title", "abstract", "label"])
    logger.info(f"Processing {filename}")
    #for debug purposes, limit to 10 rows
    #connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession() as session:
        tasks = [
            preprocess_data(session, paper,semaphores)
            for _, paper in df.iterrows()
            if pd.notna(paper.get("label_included"))
        ]

        results = await tqdm_async.gather(*tasks, desc=f"Processing {filename}")
        await session.close()
    logger.info(f"Finished processing {filename}")
    for result in results:
        if result:
            outputDF = pd.concat([outputDF, pd.DataFrame([result])], ignore_index=True)

    outputPathCSV = os.path.join(processedDataPath, f"{filename}_processed.csv")
    outputPathPickle = os.path.join(processedDataPath, f"{filename}_processed.pkl")
    outputDF.to_csv(outputPathCSV, index=False)
    outputDF.to_pickle(outputPathPickle)
    logger.info(f"Processed data saved to {filename}_processed.csv")
    
    

# Process all files in the dataset path
async def download_data():
    #create the folder for the processed datasets
    os.makedirs(processedDataPath, exist_ok=True)
    for file in os.listdir(datasetPath):
        if file.endswith("_ids.csv"):
            # Create a new event loop for each dataset
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create semaphores within the loop
            semaphores = {
                "openalex": asyncio.Semaphore(RATE_LIMITS["openalex"]["limit"]),
                "eutils": asyncio.Semaphore(RATE_LIMITS["eutils"]["limit"]),
                "crossref": asyncio.Semaphore(RATE_LIMITS["crossref"]["limit"]),
            }

            try:
                # Run the processing for the current dataset
                logger.debug(f"Processing {file}")
                await process_file(file, semaphores)
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
            


if __name__ == "__main__":
    # Change this to display only info messages and so on
    console_handler.setLevel(logging.INFO)
    clone_dataset_repo()
    #download_data()
    asyncio.run(download_data())