import subprocess
import sys

'''
The provided Python function, install(name), is a utility designed to simplify the installation of Python packages using
the pip package manager. The function takes a single parameter, name, which represents the name of the package to be in-
-stalled.
'''

def install(name):
  subprocess.call([sys.executable, '-m', 'pip', 'install', name])

lib_list = ['bs4', 'requests', 'selenium', 'chromium-chromedriver']
for libs in lib_list:
    install(libs)


# Importing all necessary libraries
import re
import time
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from google.colab import drive
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
    
'''
selenium_setup(), is a utility function designed to set up a headless Chrome WebDriver using
the Selenium library.
'''

def selenium_setup():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    return driver


'''
This function serves the purpose of initializing an empty DataFrame with pre-defined column names. The specific columns 
of the DataFrame is intended to store information related to movies or TV shows as the infomation format for both is 
simmillar.
'''

def create_DataFrame():
    df = pd.DataFrame(columns=['Title',
                               'Release_year',
                               'Type',
                               'Streaming_platform',
                               'IMDB_ratings',
                               'Total_reviews',
                               'Genre',
                               'Runtime/Duration(mins)',
                               'Age_rating',
                               'Production_country',
                               'url']
                      )
    return df


'''
The function extracts the text content of the <h1> tag within the <body> of the HTML represented by the BeautifulSoup o-
-bject (soup). The extracted title is stripped of leading and trailing whitespaces. The function then updates the 
'Title' key in the provided dictionary (_dict) with the extracted title.
'''

def get_title(soup, _dict):
    _dict['Title'] = soup.body.h1.text.strip()
    return _dict


'''
The function uses soup.find() to locate a <span> element with the class 'text-muted' within the HTML structure. It then 
retrieves the text content of that <span> element. The regular expression re.match(r'\((.*?)\)', ...), is used to extra-
-ct the content within parentheses, which is assumed to represent the release year.
'''

def get_year(soup, _dict):
    _dict['Release_year'] = re.match(r'\((.*?)\)', soup.find('span', class_='text-muted').text.strip()).group(1)
    return _dict


"""
The function uses a regular expression (genre_regex) to match a specific pattern within the HTML structure. This pattern
appears to be related to genre information and is expected to be found within a div with certain class attributes. If 
the regular expression finds a match within the HTML content represented by main_tag[ele], it extracts the genre inform-
-ation using genre_regex.match(...).group(1). The genre information is then cleaned using re.sub to replace occurrences 
of '&amp;' with 'and'.
"""

def get_genre(main_tag, ele, _dict):
    genre_regex = re.compile(
        r'<div class="detail-infos"><h3 class="detail-infos__subheading">Genres</h3><div class="detail-infos__value">(.*?)</div></div>')
    if genre_regex.search(str(main_tag[ele])):
        genre_ = genre_regex.match(str(main_tag[ele])).group(1)
        _dict['Genre'] = re.sub(r'&amp;', 'and', genre_)
    return _dict


'''
The function uses two regular expressions (runtime_regex1 and runtime_regex2) to match two possible patterns within the 
HTML structure related to runtime information. These patterns are expected to be found within a div with certain class 
attributes. If runtime_regex1 finds a match within the HTML content represented by main_tag[ele], it extracts the hours 
and minutes separately and calculates the total duration in minutes. If runtime_regex2 finds a match, it extracts the d-
-uration in minutes directly.
'''

def get_runtime(main_tag, ele, _dict):
    runtime_regex1 = re.compile(
        r'<div class="detail-infos"><h3 class="detail-infos__subheading">Runtime</h3><div class="detail-infos__value">(.*?)h (.*?)min</div></div>')
    runtime_regex2 = re.compile(
        r'<div class="detail-infos"><h3 class="detail-infos__subheading">Runtime</h3><div class="detail-infos__value">(.*?)min</div></div>')
    if runtime_regex1.search(str(main_tag[ele])):
        _dict['Runtime/Duration(mins)'] = 60 * int(runtime_regex1.match(str(main_tag[ele])).group(1)) + int(
            runtime_regex1.match(str(main_tag[ele])).group(2))
    elif runtime_regex2.search(str(main_tag[ele])):
        _dict['Runtime/Duration(mins)'] = int(runtime_regex2.match(str(main_tag[ele])).group(1))
    return _dict


'''
The function uses a regular expression (imdb_regex) to match a specific pattern within the HTML structure related to 
IMDb rating and total reviews information. This pattern is expected to be found within a span element with certain attr-
-ibutes. If the regular expression finds a match within the HTML content represented by main_tag[ele], it extracts the 
IMDb rating and total reviews separately.
'''

def get_IMDb_rating(main_tag, ele, _dict):
    imdb_regex = re.compile(r'tooltip="IMDB"/><span>\s(.*?)\s\((.*?)\)\s</span>')
    if imdb_regex.search(str(main_tag[ele])):
        _dict['IMDB_ratings'] = imdb_regex.search(str(main_tag[ele])).group(1).strip()
        _dict['Total_reviews'] = imdb_regex.search(str(main_tag[ele])).group(2).strip()
    return _dict


'''
The function uses a regular expression (age_regex) to match a specific pattern within the HTML structure related to age 
rating information. This pattern is expected to be found within a div with certain class attributes. If the regular exp-
-ression finds a match within the HTML content represented by main_tag[ele], it extracts the age rating information 
using age_regex.match(...).group(1).
'''

def get_age(main_tag, ele, _dict):
    age_regex = re.compile(
        r'<div class="detail-infos"><h3 class="detail-infos__subheading">Age rating</h3><div class="detail-infos__value">(.*?)</div></div>')
    if age_regex.search(str(main_tag[ele])):
        _dict['Age_rating'] = age_regex.match(str(main_tag[ele])).group(1)
    return _dict


'''
The function uses a regular expression (prod_regex) to match a specific pattern within the HTML structure related to 
production country information. This pattern is expected to be found within a div with certain class attributes. If the 
regular expression finds a match within the HTML content represented by main_tag[ele], it extracts the production count-
-ry information using prod_regex.match(...).group(1).
'''

def get_country(main_tag, ele, _dict):
    prod_regex = re.compile(
        r'<div class="detail-infos"><h3 class="detail-infos__subheading" style="max-width: fit-content"> Production country </h3><div class="detail-infos__value">(.*?)</div></div>')
    if prod_regex.search(str(main_tag[ele])):
        _dict['Production_country'] = prod_regex.match(str(main_tag[ele])).group(1)
    return _dict


'''
Finds the HTML element containing streaming service information using soup.find("div", class_="buybox-row stream"). Uses 
a try and except block to handle cases where there are no streaming services available (to catch exceptions). Iterates 
over the found links inside the stream_services element using for ele in range(len(stream_services.find_all('a'))).
Checks if a specific pattern related to streaming services (using a regular expression with re.compile) is present with-
-in the HTML content of the link. If the pattern is found, extracts the streaming service name and appends it to the 
stream_list.
'''

def get_streams(soup, _dict):
    stream_list = []
    stream_services = soup.find("div", class_="buybox-row stream")
    # Try and except to catch movies with no streaming services available
    try:
        for ele in range(len(stream_services.find_all('a'))):
            if re.compile(r'<img alt="(.*?)"').search(str(stream_services.find_all('a')[ele])):
                stream_list.append(
                    re.compile(r'<img alt="(.*?)"').search(str(stream_services.find_all('a')[ele])).group(1).strip())
    except Exception:
        return None
    _dict['Streaming_platform'] = ', '.join(stream_list)
    return _dict


"""
Sends an HTTP GET request to the provided URL using requests.get(url). Parses the HTML content of the page using Beauti-
-fulSoup with the 'html.parser'. Calls the above functions to extract specific details:
"""

def get_MovieSeries_details(url, _dict):
    # Sending an HTTP GET request to the URL
    page = requests.get(url)
    # Parsing the HTML content using BeautifulSoup with the 'html.parser'
    soup = BeautifulSoup(page.text, 'html.parser')
    # Extracts the title of the movie or series.
    get_title(soup, _dict)
    # Extracts the release year.
    get_year(soup, _dict)
    # Extracts streaming platform information.
    get_streams(soup, _dict)
    # Getting all the element of the info panel and iterating over each element to get desired data
    elements_ = soup.find_all('div', class_="detail-infos")
    for ele in range(len(elements_) // 2):
        # Extracts genre information.
        get_genre(elements_, ele, _dict)
        # Extracts IMDb rating and total reviews information.
        get_IMDb_rating(elements_, ele, _dict)
        # Extracts age rating information
        get_age(elements_, ele, _dict)
        # Extracts production country information.
        get_country(elements_, ele, _dict)
        # Extracts runtime information.
        get_runtime(elements_, ele, _dict)
    # return updated df
    return _dict


'''
The get_Movies_ function aim at scraping information about movies from a given URL and appending the details to
a DataFrame.
'''

def get_Movies_(url, df):
    # HTTP request
    page = requests.get(url)
    # Parsing the HTML content using BeautifulSoup with the 'html.parser'
    soup = BeautifulSoup(page.text, 'html.parser')
    # Iterates over all the links (<a> tags with an href attribute) within a specific div with class "title-list-grid."
    for ele in soup.find("div", {"class": "title-list-grid"}).find_all("a", href=True):
        # For each movie link, constructs the full movie URL by appending the href to the base URL ("https://www.justwatch.com")
        movie_url = "https://www.justwatch.com" + ele['href']
        # Initializes a dictionary (movie_ele_dict) to store movie details with default values
        movie_ele_dict = {
            'Title': None,
            'Release_year': None,
            'Type': "movie",                                                                           # Note for movies
            'Streaming_platform': None,
            'IMDB_ratings': None,
            'Total_reviews': None,
            'Genre': None,
            'Runtime/Duration(mins)': None,
            'Age_rating': None,
            'Production_country': None,
            'url': movie_url
        }
        # Calls the get_MovieSeries_details function to extract details for the current movie URL and updates the dictionary
        # Appends the dictionary as a new row to the provided DataFrame (df)
        df.loc[len(df)] = get_MovieSeries_details(movie_url, movie_ele_dict)
    # Returns the updated DataFrame
    return df


'''
The get_Series_ function seems to be designed to collect information about TV series from a given URL and append the de-
-tails to a DataFrame.
'''

def get_Series_(url, df):
    # HTTP request
    page = requests.get(url)
    # Parsing the HTML content using BeautifulSoup with the 'html.parser'
    soup = BeautifulSoup(page.text, 'html.parser')
    # Interating over all the links in the grid to send request to every single link
    # and getting their details
    for ele in soup.find("div", {"class": "title-list-grid"}).find_all("a", href=True):
        series_url = "https://www.justwatch.com" + ele['href']
        series_ele_list = {
            'Title': None,
            'Release_year': None,
            'Type': "series",                                                                          # Note for series
            'Streaming_platform': None,
            'IMDB_ratings': None,
            'Total_reviews': None,
            'Genre': None,
            'Runtime/Duration(mins)': None,
            'Age_rating': None,
            'Production_country': None,
            'url': series_url
        }
        # Calls the get_MovieSeries_details function to extract details for the current TV series URL and updates the dictionary
        # Appends the dictionary as a new row to the provided DataFrame (df)
        df.loc[len(df)] = get_MovieSeries_details(series_url, series_ele_list)
    # Returns the updated DataFrame
    return df


'''
The start_scraping function is the main function responsible for initiating the web scraping process to gather informat-
-ion about movies and TV shows.
'''

# Main function to get movies and film data
def start_scraping():
    '''
    Although using the selenium driver is adjunct here, as we can extract our data directly using the url.
    However, I have added it so if we want to say use the filter or sort buttons in the website, we can
    call the driver and navigate those components. Selenium Webdriver is not called in the project as
    it gave an error in collab regarding invalid session!
    
    Note: If you are using the driver, DO REMEMBER TO CLOSE IT as it consume a lot of memory!
    '''
    # driver = 
    # Calls the _movies function to extract movie details and stores them in a DataFrame (df_movies_).
    def _movies():
        print("Extracting movies...")
        df_movies_ = create_DataFrame()
        url_movies = 'https://www.justwatch.com/in/movies?page='
        for i in range(1, 5):
            url_current = url_movies + str(i)
            get_Movies_(url_current, df_movies_)
        print("Movies extracted!")
        return df_movies_
    # Calls the _series function to extract TV show details and stores them in another DataFrame (df_series_).
    def _series():
        print("Extracting tv-shows...")
        df_series_ = create_DataFrame()
        url_series = 'https://www.justwatch.com/in/tv-shows?page='
        for i in range(1, 5):
            url_current = url_series + str(i)
            get_Series_(url_current, df_series_)
        print("Tv-shows extracted!")
        return df_series_
    # Concatenates the movie and TV show DataFrames into a single DataFrame (combined).
    combined = pd.concat([_movies(), _series()], ignore_index=True)
    # Returns the combined DataFrame containing information about both movies and TV shows.
    return combined


"""
The filter_data function filter a DataFrame containing information about movies and TV shows based on certain criteria:

- The filtering criteria include movies and TV shows released in the last 2 years,
- and with an IMDb rating greater than 7.0
"""

def filter_data(df_origin):
    df = df_origin.copy(deep=True)
    # Only include movies and TV shows released in the last 2 years (from the current date).
    # The lambda function get_year is used to dynamically calculate the year two years ago from the current date.
    get_year = lambda: (datetime.now() - timedelta(days=365 * 2)).strftime("%Y")
    year_ = get_year()
    # Only include movies and TV shows released with the last 2 years
    df = df[df['Release_year'] > year_]
    # Converts the 'IMDB_ratings' column to float type to enable numerical comparison.
    df['IMDB_ratings'] = df.loc[:, ('IMDB_ratings')].astype(float)
    # Only include movies and TV shows with an IMDb rating of 7 or higher.
    df = df[df['IMDB_ratings'] > 7.0]
    # Resets the index of the DataFrame and drops the old index.
    df.reset_index(inplace=True, drop=True)
    # return df
    return df


"""
The get_info function provide summary information about a DataFrame containing details about movies and TV shows. 
"""

def get_info(df):
    # The total number of rows (elements) in the DataFrame
    print(f'Total elements(rows): {df.shape[0]}')
    # The total number of columns (attributes) in the DataFrame.
    print(f'Total attributes(columns): {df.shape[1]}')
    # The number of duplicated rows in the DataFrame.
    print(f'Duplicated items: {df.duplicated().sum()}')
    # The number of null values in the 'IMDB_ratings' column.
    print(f'IMDB ratings null items: {df.IMDB_ratings.isna().sum()}')
    # The number of null values in the 'Genre' column.
    print(f'Genre null items: {df.Genre.isna().sum()}')
    # The number of null values in the 'Streaming_platform' column, with a note indicating that these represent shows
    # and movies that cannot be watched on an OTT (Over-The-Top) platform.
    print(f'Streaming_platform null items: {df.Streaming_platform.isna().sum()} (shows and movie that cannot be watched in an OTT)\n')


"""
The data_analysis function perform some basic data analysis on a DataFrame containing information about movies and TV 
shows. 
"""

def data_analysis(df, is_origin=True):
    if is_origin:
      df['IMDB_ratings'] = df.loc[:, ('IMDB_ratings')].astype(float)
    df_type_ = df.groupby(['Type'])
    # The total number of TV series in the DataFrame.
    print(f"Total number of tv-series in the dataframe: {len(df_type_.get_group('series'))}")
    # The total number of movies in the DataFrame.
    print(f"Total number of movies in the dataframe: {len(df_type_.get_group('movie'))}")
    # The average IMDb rating for movies in the DataFrame.
    print(f"Average IMDb rating for the movies: {float(df_type_.get_group('movie').agg({'IMDB_ratings': 'mean'})):.2f}")
    # The average IMDb rating for TV shows in the DataFrame.
    print(f"Average IMDb rating for the TV shows: {float(df_type_.get_group('series').agg({'IMDB_ratings': 'mean'})):.2f}")


"""
The get_top_genres function analyze the 'Genre' column in the filtered or original DataFrame containing information abo-
-ut movies and TV shows to identify and visualize the top genres.
"""

#  Identify the top 5 genres that have the highest number of available movies and TV shows.
def get_top_genres(df, is_origin=True):
    # Initializes an empty dictionary (genre_dict) to store genre counts.
    genre_dict = dict()
    # Iterates through each element in the 'Genre' column of the DataFrame
    for ele in df.Genre:
        # Splits the genres into a list using commas as separators.
        my_list = ele.split(',')
        for sub_ele in my_list:
            # If the genre is already in the dict
            if sub_ele.strip() in genre_dict:
                # Increments the count for each genre in the genre_dict.
                genre_dict[sub_ele.strip()] += 1
            # If the genre not in already in the dict
            else:
                # Add first occurance in the dict
                genre_dict[sub_ele.strip()] = 1
    # Sorts the genre_dict based on genre counts in descending order.
    sorted_dict = dict(sorted(genre_dict.items(), key=lambda item: item[1], reverse=True))
    print("\nTop Genres: ")
    print(sorted_dict)
    # Creating a bar chart to better visualize the data
    labels = list(sorted_dict.keys())
    values = list(sorted_dict.values())

    # Plotting the pie chart
    plt.figure(figsize=(20, 4))
    plt.bar(labels, values, color='red')

    # Adding a title and labels
    if is_origin:
      plt.title('Top genres for Orginal Dataset')
    else:
      plt.title('Top genres for Filtered Dataset')
    plt.xlabel('genres')
    plt.xticks(rotation=45)
    plt.ylabel('available movies & tv-shows')


"""
The get_top_stream function analyze the 'Streaming_platform' column in the filtered or original DataFrame containing in-
-formation about movies and TV shows to identify and visualize the top genres.
"""

# Determine the streaming service with the most significant number of offerings.
def get_top_stream(df, is_origin=True):
    # Initializes an empty dictionary (stream_dict) to store streaming platform counts.
    stream_dict = dict()
    # Iterates through each element in the 'Streaming_platform' column of the DataFrame.
    for ele in df.Streaming_platform:
        # Skips iterations where the element is None.
        if ele == None:
            continue
        else:
            # Splits the streaming platforms into a list using commas as separators.
            my_list = ele.split(',')
            for sub_ele in my_list:
                # If the streaming platform is in already in the dict
                if sub_ele.strip() in stream_dict:
                    # Increments the count for each streaming platform in the stream_dict.
                    stream_dict[sub_ele.strip()] += 1
                # If the streaming platform not in already in the dict
                else:
                    stream_dict[sub_ele.strip()] = 1

    sorted_dict = dict(sorted(stream_dict.items(), key=lambda item: item[1], reverse=True))
    print("Top Streaming Platforms: ")
    print(sorted_dict)

    # Creating a bar chart to better visualize the data
    labels = list(sorted_dict.keys())
    values = list(sorted_dict.values())

    # Plotting the pie chart
    plt.figure(figsize=(20, 4))
    plt.bar(labels, values, color='cyan')

    # Adding a title and labels
    if is_origin:
      plt.title('Top streaming platforms for Orginal Dataset')
    else:
      plt.title('Top streaming platforms for Filtered Dataset')
    plt.xlabel('streaming platforms')
    plt.xticks(rotation=45)
    plt.ylabel('number of offerings')


'''
The start_filter_and_analysis function perform a series of filtering and analysis tasks on a DataFrame containing infor-
-mation about movies and TV shows.
'''

# Main function for data analysis!
def start_filter_and_analysis(df_original):
    # Calls the filter_data function to filter the original dataset to only include movies and TV shows released in the
    # last 2 years (from the current date) and with an IMDb rating of 7 or higher. The filtered DataFrame is stored in
    # df_fil.
    df_fil = filter_data(df_original)
    print("Filtered Dataset created")
    print("\nFor Original Dataset---------------------------->")
    # Calls the get_info function to print basic information about the original dataset, including the total number of
    # elements, attributes, duplicated items, IMDb ratings null items, genre null items, and streaming platform null it-
    # -ems.
    get_info(df_original)
    # Calls the data_analysis function to print an analysis of the original dataset, including the total number of movi-
    # -es and TV shows, and average IMDb ratings for movies and TV shows.
    data_analysis(df_original, is_origin=True)
    # Calls the get_top_genres function to visualize and print the top genres in the original dataset.
    get_top_genres(df_original, is_origin=True)
    # Calls the get_top_stream function to visualize and print the top streaming platforms in the original dataset.
    get_top_stream(df_original)
    print("\nFor Filtered Dataset---------------------------->")
    # Calls the get_info function to print basic information about the filtered dataset, including the total number of
    # elements, attributes, duplicated items, IMDb ratings null items, genre null items, and streaming platform null it-
    # -ems.
    get_info(df_fil)
    # Calls the data_analysis function to print an analysis of the filtered dataset, including the total number of movi-
    # -es and TV shows, and average IMDb ratings for movies and TV shows.
    data_analysis(df_fil, is_origin=False)
    # Calls the get_top_genres function to visualize and print the top genres in the filtered dataset.
    get_top_genres(df_fil, is_origin=False)
    # Calls the get_top_stream function to visualize and print the top streaming platforms in the filtered dataset.
    get_top_stream(df_fil)
    print("\n")
    # Returns the filtered DataFrame (df_fil).
    return df_fil

'''
convert_to_csv function converts the dataframe into a .csv format
'''

def convert_to_csv(df, file_name):
    csv_file = df.to_csv(file_name, index=False)
    print(f"DataFrame successfully converted and saved as {file_name}")
    return csv_file

# Main function to run the script
def main():
  print("JustWatch popular movie scraper started.....\n")
  time.sleep(1)
  df_original = start_scraping()
  print("\nOriginal Dataset created")
  df_filtered = start_filter_and_analysis(df_original)

  # Convert the original dataframe to csv format
  convert_to_csv(df_original, "justwatch_original.csv")
  # Convert the filtered dataframe to csv format
  convert_to_csv(df_filtered, "justwatch_filtered.csv")


'''
# To run the file as .py in your local machine
if __name__ == "__main__":
    lib_list = ['bs4', 
                'requests',
                'selenium', 
                'chromium-chromedriver', 
                'pandas',
                'numpy']
    for libs in lib_list:
        install(libs)

    main()
'''

'''
# To run on google colab, save this file as in .py format in your MyDrive
# Open a new notebook, then import and mount colab drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate where you have saved the .py file
%cd /content/drive/MyDrive/<your directory>/

# Import the .py file and call main()
from just_watch_webscarper import main
# call the function to execute
main()
'''

'''
Dataset Drive Link (View Access with Anyone)-

Original dataset: https://drive.google.com/file/d/1-Jvz-_k265An32PmXakjn-P4BChVpE03/view?usp=sharing
Filtered dataset: https://drive.google.com/file/d/1-FQhXQ-R0LqTNe9wav2JE_thcqRD_zUZ/view?usp=sharing
'''
