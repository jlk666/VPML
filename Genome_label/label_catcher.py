import requests
from bs4 import BeautifulSoup

def get_isolation_source(biosample_id):
    efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=biosample&id={biosample_id}"
    response = requests.get(efetch_url)
    soup = BeautifulSoup(response.text, 'lxml') # Use 'lxml' parser for XML
    
    attributes = soup.find('attributes')
    for attribute in attributes.find_all('attribute'):
        if attribute['attribute_name'].lower() == 'isolation source':
            return attribute.text
    
    return None

biosample_id = 'SAMN02741361'
print(get_isolation_source(biosample_id))
