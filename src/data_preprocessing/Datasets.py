import pandas as pd
import requests


Calcium = pd.read_csv("./Datasets/Raw/Cohen_2006_CalciumChannelBlockers_ids.csv")
Depression = pd.read_csv("./Datasets/Raw/Bannach-Brown_2019_ids.csv")
Virus = pd.read_csv("./Datasets/Raw/Kwok_2020_ids.csv")

print(Calcium.isna().sum()) #pmid has no missing values -> we will retrieve from pmid
Calcium.head()

print(Depression.isna().sum())
Depression.head()

print(Virus.isna().sum()) 
Virus.head()

# Define functions to get metadata from APIs

#Fetches metadata from PubMed using PMID.
def get_pubmed_metadata(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=xml"
    response = requests.get(url)
    if response.status_code == 200:
        # Parsing XML response to extract title and abstract
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.text)
        doc = root.find(".//DocSum")
        title = doc.find(".//Item[@Name='Title']").text if doc is not None else None
        abstract = doc.find(".//Item[@Name='Source']").text if doc is not None else None
        return title, abstract
    else:
        return None, None

#Fetches metadata from CrossRef using DOI.
def get_crossref_metadata(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Safely get the title, handling empty lists
        title_list = data["message"].get("title", ["No title available"])
        title = title_list[0] if title_list else "No title available"  # Use the first element or default
        abstract = data["message"].get("abstract", "No abstract available")
        return title, abstract
    else:
        return None, None


#Fetches metadata from OpenAlex using OpenAlex ID.
def get_openalex_metadata(openalex_id):
    url = f"https://api.openalex.org/works/{openalex_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        title = data["title"] if "title" in data else None
        abstract = data.get("abstract", "No abstract available")
        return title, abstract
    else:
        return None, None

def preprocess_data(paper):
    """
    Tries to fetch metadata based on available identifiers (PMID, DOI, OpenAlex).
    
    Parameters:
        paper: A row from the DataFrame containing identifiers (PMID, DOI, OpenAlexID)
    
    Returns:
        A dictionary with title, abstract, and label.
    """
    title, abstract = None, None
    
    # Try PMID
    if pd.notna(paper.get("pmid")):
        title, abstract = get_pubmed_metadata(paper["pmid"])
    
    # If PMID doesn't exist or no data was found, try DOI
    if title is None and pd.notna(paper.get("doi")):
        title, abstract = get_crossref_metadata(paper["doi"])

    # If DOI also doesn't exist or no data was found, try OpenAlexID
    if title is None and pd.notna(paper.get("openalex_id")):
        title, abstract = get_openalex_metadata(paper["openalex_id"])
    
    # Return a dictionary if metadata was found
    if title and abstract:
        return {"title": title, "abstract": abstract, "label": paper["label_included"]}
    else:
        return None


# Prepare an output DataFrame
output_Calcium = pd.DataFrame(columns=["title", "abstract", "label"])

# Iterate through the dataset to fetch metadata
for _, paper in Calcium.iterrows():
    outputrow = preprocess_data(paper)
    if outputrow:
        output_Calcium = pd.concat([output_Calcium, pd.DataFrame([outputrow])], ignore_index=True)

# Save the resulting data to a CSV or Pickle file
output_Calcium.to_csv("processed_metadata.csv", index=False)
output_Calcium.to_pickle("processed_metadata.pkl")


# Set display options to show full content of cells
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)   # Display all columns
output_Calcium.head()