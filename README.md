# web_scraper__justwatch.com

* Website: https://www.justwatch.com/in/movies?release_year_from=2000

# Description:
JustWatch is a popular platform that allows users to search for movies and TV shows across multiple streaming services like Netflix, Amazon Prime, Hulu, etc. For this assignment, you will be required to scrape movie and TV show data from JustWatch using Selenium, Python, and BeautifulSoup. Extract data from HTML, not by directly calling their APIs. Then, perform data filtering and analysis using Pandas, and finally, save the results to a CSV file.

# Tasks:
## Web Scraping:
  Use BeautifulSoup to scrape the following data from JustWatch:

  **a. Movie Information:**
    *  Movie title \n
    * Release year \n
    * Genre
    * IMDb rating
    * Streaming services available (Netflix, Amazon Prime, Hulu, etc.)
    * URL to the movie page on JustWatch
  
  **b. TV Show Information:**
    * TV show title
    * Release year
    * Genre
    * IMDb rating
    * Streaming services available (Netflix, Amazon Prime, Hulu, etc.)
    * URL to the TV show page on JustWatch
      
  **c. Scope:**
     * Scrape data for at least 50 movies and 50 TV shows.
     * You can choose the entry point (e.g., starting with popular movies,
       or a specific genre, etc.) to ensure a diverse dataset.`

## Data Filtering & Analysis:
  After scraping the data, use Pandas to perform the following tasks:

  **a. Filter movies and TV shows based on specific criteria:**
     * Only include movies and TV shows released in the last 2 years (from the current date).
     * Only include movies and TV shows with an IMDb rating of 7 or higher.
       
  **b. Data Analysis:**
     * Calculate the average IMDb rating for the scraped movies and TV shows.
     * Identify the top 5 genres that have the highest number of available movies and TV shows.
     * Determine the streaming service with the most significant number of offerings.
   
## Data Export:
    * Dump the filtered and analysed data into a CSV file for further processing and reporting.

# Note:
The respo. contains the following files:
* **Web scraper notebook:** contains a googlecollab notebook to extract the info. directly in the notebook.
* **Original dataset:** contains the original dataset of the extracted data.
* **Filtered dataset:** contains the filtered dataset based on the above criteria(s).
* **Web scraper script:** contains a .py script to run the program on your system
     
