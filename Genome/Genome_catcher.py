import requests
from bs4 import BeautifulSoup
import csv
import urllib.request
import os

def download_file(url, destination_folder, number_downloaded):
    # Create the desired folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    filename = os.path.join(destination_folder, os.path.basename(url))


    try:
        urllib.request.urlretrieve(url, filename)
        number_downloaded = number_downloaded + 1
        print("File downloaded successfully.")
        return number_downloaded
    
    except Exception as e:
        print(f"An error occurred during the download: {str(e)}")
        print("this url failed", url)
        return number_downloaded

def URL_from_CSV(file_path):
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)

        # Skip the header row (if it exists)
        next(csv_reader)

        # Read the values from the second column and convert them into a list
        genebank_url_list = [row[15] for row in csv_reader]

        #convert FTP to HTTP
        trimmed_URL = []
        for URL in genebank_url_list:
            URL_HTTP_link =  convert_ftp_to_http(URL)
            trimmed_URL.append(URL_HTTP_link)
            trimmed_URL = list(set(trimmed_URL)) #remove empty url
            trimmed_URL = [elem for elem in trimmed_URL if elem != '/']


    return trimmed_URL

def convert_ftp_to_http(url):
    # Replace "ftp://" with "http://"
    http_url = url.replace("ftp://", "http://")

    # Add a trailing slash if not present
    if not http_url.endswith("/"):
        http_url += "/"

    return http_url


def genome_catch(url, destination_folder, number_downloaded):
    
    id = url.strip().split("/")[-2]

    # Send a GET request to fetch the HTML content
    response = requests.get(url)
    html_content = response.text

    # Create a BeautifulSoup object to parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all <a> tags (links) in the HTML
    links = soup.find_all("a")

    # Extract the href attribute from each <a> tag
    urls = [link.get("href") for link in links] 

    # Extract the genomic data file only 
    filtered_urls = [url for url in urls if "_genomic.fna.gz" in url]
    genome_url = min(filtered_urls, key=len)

    #concante the url 
    url = f"https://ftp.ncbi.nlm.nih.gov{urls[0]}/{id}/{genome_url}"
    

    number_downloaded = download_file(url, destination_folder, number_downloaded)
    #print(number_downloaded)
    
    return number_downloaded

def GenomeCatcher(csv_file, outputdir):
    number_downloaded = 0
    http_link = URL_from_CSV(csv_file)
    for URL in http_link:
        number_downloaded = genome_catch(URL, outputdir,number_downloaded)
    

GenomeCatcher('prokaryotes.csv','genome') 

