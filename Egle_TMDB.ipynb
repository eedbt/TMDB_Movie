{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8a97057b-db51-4939-b4a8-ff964025c9e0",
   "metadata": {},
   "source": [
    "# Analysis of TMDB Movie dataset\n",
    "### Contents\n",
    "- Data cleaning and management\n",
    "- Can budgetless movies sell?\n",
    "- What is the dispersion between women and non-women directed movies?\n",
    "- Are woman-directed movies less popular?\n",
    "- Do women directed movies receive less budget? What about revenue?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "47561dcf-a634-49b5-b950-254dc32c5303",
   "metadata": {},
   "source": [
    "## 1. Data cleaning and management"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "666be85d-d740-4cbb-b63a-da1275e4701a",
   "metadata": {},
   "source": [
    "### Importing and loading the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "790156f0-06e7-4b7a-82da-33bab16a452a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np  # Import the numpy library for numerical operations\n",
    "import seaborn as sns  # Import the seaborn library for data visualization\n",
    "import matplotlib.pyplot as plt  # Import the pyplot module from matplotlib for plotting\n",
    "import pandas as pd  # Import the pandas library for data manipulation and analysis\n",
    "import ast  # Import the ast module for working with literal_eval functions\n",
    "\n",
    "# Define the file name\n",
    "movies = \"tmdb_movies.csv\"\n",
    "\n",
    "# Read the CSV file into a DataFrame\n",
    "df = pd.read_csv(movies)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14b93f6f-e308-4353-9227-25dd35882fef",
   "metadata": {},
   "source": [
    "### Extracting values from dictionary columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0898ee67-cf38-49de-9b9c-d0d1d7aee635",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to extract genres for each datapoint\n",
    "def extract_genres(x): \n",
    "    x = ast.literal_eval(x) # Transform '['name', 'id']' back into ['name', 'id']\n",
    "    Genres = [] # Empty list to store the genres\n",
    "    \n",
    "#Iterate through each dictionary\n",
    "    for item in x: # Iterate for each dictionary in our list\n",
    "        Genres.append(item['name']) # Grab the 'name' key for each dictionary\n",
    "    \n",
    "    return Genres #Return the Genres\n",
    "\n",
    "# Apply the function to genres\n",
    "df['extracted_genres'] = df['genres'].apply(extract_genres)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ede523d4-fae2-439c-8ef5-8f0e67c02926",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Function to extract production companies for each datapoint\n",
    "def extract_prod_comp(x): \n",
    "    x = ast.literal_eval(x) # Transform '['name', 'id']' back into ['name', 'id']\n",
    "    Production_Companies = [] # Empty list to store the production companies\n",
    "    \n",
    "#Iterate through each dictionary \n",
    "    for item in x: # iterate for each dictionary in our list\n",
    "        Production_Companies.append(item['name']) # Grab the 'name' key for each dictionary\n",
    "        \n",
    "    return Production_Companies #Return the Production Companies\n",
    "\n",
    "# Apply the function to production companies\n",
    "df['extracted_production_companies'] = df['production_companies'].apply(extract_prod_comp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "23dbdc0a-a3c7-475a-bc2d-eb2c03947d51",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to extract production countries for each datapoint\n",
    "def extract_prod_countries(x): \n",
    "    x = ast.literal_eval(x) # Transform '['name', 'id']' back into ['name', 'id']\n",
    "    Production_Countries = [] # Empty list to store the production countries\n",
    "    \n",
    "#Iterate through each dictionary\n",
    "    for item in x: # iterate for each dictionary in our list\n",
    "        Production_Countries.append(item['name']) # Grab the 'name' key for each dictionary\n",
    "    \n",
    "    return Production_Countries #Return the Production Countries\n",
    "\n",
    "# Apply the function to production countries\n",
    "df['extracted_production_countries'] = df['production_countries'].apply(extract_prod_countries)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d1b29c0a-d119-4f31-9022-d19dbb52ea03",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to extract keywords for each datapoint\n",
    "def extract_keywords(x): \n",
    "    x = ast.literal_eval(x) # Transform '['name', 'id']' back into ['name', 'id']\n",
    "    Keywords = [] # Empty list to store the keywords\n",
    "    \n",
    "#Iterate through each dictionary \n",
    "    for item in x: # iterate for each dictionary in our list\n",
    "        Keywords.append(item['name']) # Grab the 'name' key for each dictionary\n",
    "        \n",
    "    return Keywords # Return the Keywords\n",
    "\n",
    "# Apply the function to keywords\n",
    "df['extracted_keywords'] = df['keywords'].apply(extract_keywords)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "9d45c0d4-cb07-44b2-b296-83700da74a30",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to extract spoken langauges for each datapoint\n",
    "def extract_spoken_languages(x): \n",
    "    x = ast.literal_eval(x) # Transform '['name', 'id']' back into ['name', 'id']\n",
    "    Spoken_Languages = [] # Empty list to store the spoken langauges\n",
    "    \n",
    "#Iterate through each dictionary \n",
    "    for item in x: # iterate for each dictionary in our list\n",
    "        Spoken_Languages.append(item['name']) # Grab the 'name' key for each dictionary\n",
    "        \n",
    "    return Spoken_Languages #Return the Spoken Langauges\n",
    "\n",
    "## Apply the function to spoken langauges\n",
    "df['extracted_spoken_languages'] = df['spoken_languages'].apply(extract_spoken_languages)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0fa618c1-7505-4829-8a7e-36017d0c3a2b",
   "metadata": {},
   "source": [
    "### Looking for duplicate movie id's"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 460,
   "id": "bf206d62-dec8-442f-9237-585879645f4b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4803"
      ]
     },
     "execution_count": 460,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.id.nunique() #Dispay number of unique movie id's"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 461,
   "id": "3e6ff5c7-dd72-4c5b-97bb-dc999318f816",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4803, 25)"
      ]
     },
     "execution_count": 461,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape #Display the number of rows and column of our dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "908e522c-badf-4356-a1f5-4e470938584f",
   "metadata": {},
   "source": [
    "##### Notice that the number of unique id's matches the number of rows. It means that all movie id's in the dataset are unique and further cleaning is not needed."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "de1080e9-282a-45e5-b407-189a25d121ef",
   "metadata": {},
   "source": [
    "### Handling  missing values"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b916bc6-07c3-4f75-b24e-93e3b5f528b9",
   "metadata": {},
   "source": [
    "##### Demonstating the number and percentage of missing values for each column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 463,
   "id": "2c6694c6-250f-44b9-bb68-12ce10799929",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>null</th>\n",
       "      <th>percent</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>homepage</th>\n",
       "      <td>3091</td>\n",
       "      <td>64.356</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>tagline</th>\n",
       "      <td>844</td>\n",
       "      <td>17.572</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>overview</th>\n",
       "      <td>3</td>\n",
       "      <td>0.062</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>runtime</th>\n",
       "      <td>2</td>\n",
       "      <td>0.042</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>release_date</th>\n",
       "      <td>1</td>\n",
       "      <td>0.021</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>budget</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>status</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>extracted_keywords</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>extracted_production_countries</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>extracted_production_companies</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>extracted_genres</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>vote_count</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>vote_average</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>revenue</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>spoken_languages</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>genres</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>production_countries</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>production_companies</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>popularity</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>original_title</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>original_language</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>keywords</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>extracted_spoken_languages</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                null  percent\n",
       "homepage                        3091   64.356\n",
       "tagline                          844   17.572\n",
       "overview                           3    0.062\n",
       "runtime                            2    0.042\n",
       "release_date                       1    0.021\n",
       "budget                             0    0.000\n",
       "status                             0    0.000\n",
       "extracted_keywords                 0    0.000\n",
       "extracted_production_countries     0    0.000\n",
       "extracted_production_companies     0    0.000\n",
       "extracted_genres                   0    0.000\n",
       "vote_count                         0    0.000\n",
       "vote_average                       0    0.000\n",
       "title                              0    0.000\n",
       "revenue                            0    0.000\n",
       "spoken_languages                   0    0.000\n",
       "genres                             0    0.000\n",
       "production_countries               0    0.000\n",
       "production_companies               0    0.000\n",
       "popularity                         0    0.000\n",
       "original_title                     0    0.000\n",
       "original_language                  0    0.000\n",
       "keywords                           0    0.000\n",
       "id                                 0    0.000\n",
       "extracted_spoken_languages         0    0.000"
      ]
     },
     "execution_count": 463,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Handling missing values\n",
    "def null_vals(dataframe):\n",
    "#Function to show both number of nulls and the percentage of nulls in the whole column\n",
    "    null_vals = dataframe.isnull().sum() # How many nulls in each column\n",
    "    total_cnt = len(dataframe) # Total entries in the dataframe\n",
    "    null_vals = pd.DataFrame(null_vals,columns=['null']) # Put the number of nulls in a single dataframe\n",
    "    null_vals['percent'] = round((null_vals['null']/total_cnt)*100,3) # Round how many nulls are there, as %, of the df\n",
    "    \n",
    "    return null_vals.sort_values('percent', ascending=False) #Return the number and percentage of missing values, order by percentage descending\n",
    "null_vals(df) #Display the table"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3d52d4f0-6333-49d4-8cb1-140948185990",
   "metadata": {},
   "source": [
    "##### Since informtion about movies' homepage, tagline, overview and runtime will not be used in this analysis, I ignore the missing values. \n",
    "##### Looking for the movie, which does not have its release date"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 464,
   "id": "ad1b2ef6-4cf2-4d26-b2ef-68a4eaec289e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>budget</th>\n",
       "      <th>genres</th>\n",
       "      <th>homepage</th>\n",
       "      <th>id</th>\n",
       "      <th>keywords</th>\n",
       "      <th>original_language</th>\n",
       "      <th>original_title</th>\n",
       "      <th>overview</th>\n",
       "      <th>popularity</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>production_countries</th>\n",
       "      <th>release_date</th>\n",
       "      <th>revenue</th>\n",
       "      <th>runtime</th>\n",
       "      <th>spoken_languages</th>\n",
       "      <th>status</th>\n",
       "      <th>tagline</th>\n",
       "      <th>title</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>extracted_genres</th>\n",
       "      <th>extracted_production_companies</th>\n",
       "      <th>extracted_production_countries</th>\n",
       "      <th>extracted_keywords</th>\n",
       "      <th>extracted_spoken_languages</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>4553</th>\n",
       "      <td>0</td>\n",
       "      <td>[]</td>\n",
       "      <td>NaN</td>\n",
       "      <td>380097</td>\n",
       "      <td>[]</td>\n",
       "      <td>en</td>\n",
       "      <td>America Is Still the Place</td>\n",
       "      <td>1971 post civil rights San Francisco seemed li...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>[]</td>\n",
       "      <td>[]</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>[]</td>\n",
       "      <td>Released</td>\n",
       "      <td>NaN</td>\n",
       "      <td>America Is Still the Place</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>[]</td>\n",
       "      <td>[]</td>\n",
       "      <td>[]</td>\n",
       "      <td>[]</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      budget genres homepage      id keywords original_language  \\\n",
       "4553       0     []      NaN  380097       []                en   \n",
       "\n",
       "                  original_title  \\\n",
       "4553  America Is Still the Place   \n",
       "\n",
       "                                               overview  popularity  \\\n",
       "4553  1971 post civil rights San Francisco seemed li...         0.0   \n",
       "\n",
       "     production_companies production_countries release_date  revenue  runtime  \\\n",
       "4553                   []                   []          NaN        0      0.0   \n",
       "\n",
       "     spoken_languages    status tagline                       title  \\\n",
       "4553               []  Released     NaN  America Is Still the Place   \n",
       "\n",
       "      vote_average  vote_count extracted_genres  \\\n",
       "4553           0.0           0               []   \n",
       "\n",
       "     extracted_production_companies extracted_production_countries  \\\n",
       "4553                             []                             []   \n",
       "\n",
       "     extracted_keywords extracted_spoken_languages  \n",
       "4553                 []                         []  "
      ]
     },
     "execution_count": 464,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df.release_date.isnull()] #Display the movie with missing values in release_date"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ecf08c6-751b-4ed2-805f-a11b76dc2512",
   "metadata": {},
   "source": [
    "##### Since there is only 1 missing value in release_date column and the data about the movie are not informative for this analysis, I remove the movie from the datset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "66e0ccd3-db57-47fb-885e-d6df749f3b80",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop rows with missing values in the 'release_date' column\n",
    "df.dropna(subset=['release_date'], inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfd0b8b5-a6d8-4bbc-b54b-498db10410b8",
   "metadata": {},
   "source": [
    "## 2. Can budgetless movies sell?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "31049c59-619e-4c3f-bd66-71d7de798dd1",
   "metadata": {},
   "source": [
    "### Ones of the most important factors describing the success of a movie are recognition and profitability. The latter is easier to measure."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a47bb94-dc8c-4db3-bf56-1c159a6bc3c5",
   "metadata": {},
   "source": [
    "### To investigate this at first I need to look at the distribution of the budget "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 474,
   "id": "8727207d-3347-49f9-b60f-89727adcc63e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWAAAAF4CAYAAABq5CO6AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAzMElEQVR4nO3deVxU9eL/8dcAAoqoobYo4U6ShWneyl1Lv2RlmAoBRte0Ra+53kgjXL5qmpXkUuit22IuIF6XXKpvZTcwIktuShlakqaYKRomoAwI5/eHP+ZKigMKc0Z4Px+PHjlnzjnzPuP4njOfOeeMxTAMAxERcTgXswOIiNRWKmAREZOogEVETKICFhExiQpYRMQkKmAREZOogGu4rKwsAgICCA4OJjg4mIEDBxIWFsYHH3xgm2fhwoVs2LDhkut57bXX+PTTTy963/nL33TTTfz++++Vypiens60adMA+O677xg3blyllr8cxcXFjB49mqCgIFasWFHmvsWLF3PTTTexdu3aMtNPnz5Np06deOqppy77cYODgzl16tRlL1/qYn+vISEhpKWlVXpdM2fOZPHixZed5dChQ4wdO/ayl6/N3MwOINXP09OT999/33b78OHDDB8+HFdXV4KCghg/frzddWzfvp22bdte9L6KLH8p+/bt4+jRowDceuutLFq06IrWVxFHjx7liy++YOfOnbi6ul5wf7NmzXj//fcZMmSIbdrHH39MvXr1ruhxz/97uFJ//nv94IMPeO655/j444+r7DEq4tdff2X//v0OfcyaQnvAtVDz5s0ZN24cb731FgBTpkyx/XnRokUMHDiQwYMHM3LkSI4dO8bKlSv5/vvveemll/jkk0+YMmUKo0aN4v777+fll18uszzAggULeOihhwgODubf//43AOvWrSuz51h6+8iRIyxatIgdO3bw3HPPsX37dh544AEAcnNzeeaZZ3jggQcYOHAgL730EmfPngXOFfXixYsJCwvj7rvvZtWqVRfd1h07dhAaGmrbpuTkZPLy8nj88cc5e/YsgwcP5uDBgxcs17NnT/bt28dvv/1mm7Z+/XoefPBB2+3y8q1evZpRo0bZ5svMzKRnz54UFxeX+YSwZs0aBg8ezKBBgxg+fDiZmZm2zEOHDmXw4MEMHjyY//u//6vQ3+vJkydp2rQpQJnn8c+38/LyGD9+PEFB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8onx66+/Gr/88kutzbN06VIjKSnJiI+PN8aOHWssW7bMtne5f/9+Y8iQIcbvv//ukL2+imQ5ceKEQ7J88sknxq5duwyr1WoUFRUZISEhxp49e2z37dy508jKyqr2HM6WxTDOHZJoGIZx+PBh48svv7RNf/bZZ42lS5caWVlZDnsNO1OW6nZVfgnXu3dv9u7dy5o1awAIDAwEwNPTk/z8fIefYutsefr37891111Hr169KCkpYfv27bi6ujJgwABatmzJ22+/bftSozZl6d27N66urri4uHD8+HGOHz9Os2bN+PDDD3n11Vd54403aN68ea3LAucOSQRo1qwZzZo1s01v2rQpjRo1qrVZqttVWcDXXHMNf/3rX1m2bBlxcXH07t2bunXrsmfPHsaMGVPr87Ru3dr254iICFxdXfn4449xc3MjJCTEYYXnbFlKj8QA8PDwwN/fnzVr1rB582bi4uJo2bJlrcxyvqKiIn744QdOnjzJmTNn+OqrrxgyZEitz1JdrsqjIEr98ccffPHFF2zevJnGjRvzyCOP0L59e+X5/4zzTstcu3YtPXr04Lrrrqv1WQB+//13unXrRosWLViyZEmZN4ranKWwsJCNGzeSkJBAs2bNGDNmTLUdlXI1ZakuV3UBlyouLi7zJZPZnCnP+cVnNmfKcvbsWV544QUeeeQR03/w1ZmylCo9KaQ6j52vKGfKUtVqRAGLXI7zTxAxmzNlEcdRAYuImMT8z8giIrWUClhExCQqYBGRCti1axeRkZGXnGfu3LkMHTqU0NBQ0tLS7K5TBSyVsn37dttpoKVeeeUV1q1bd0XrnThxItu3b6/0cpmZmXb/UVxKcXExI0eOJDw8vMypwVOmTKFLly62b+ABdu/ezU033cT27dtJTk5m9erVZGVlERoaCpy7apjVauWNN94gPT39sjNdzPmPUyo+Pp7FixcD534r7a9//SuPPfYYjz76KBs3bgRg3bp19OnTh8jISIYNG8YjjzxCampqlWarDd58801iYmKwWq3lzrNnzx6+/fZb1qxZw0svvcQLL7xgd71X5YkYIlUlOzubnJyci76BNG3alOTkZPr16wfApk2bbJev7NWrF3CuGP/sySefrMbEFzdjxgzef/99GjRoQF5eHsHBwXTv3h2ABx54gGeeeQaA48ePM2zYMFasWEHTpk0dnvNq5efnx+LFi3n22WeBc9dTmT17NgCNGjVizpw5XHvttXh6elJYWEheXh5ubvbrVQUsVWb79u0kJCTw6quvAtC9e3dSUlKYMmUK7u7uHD58mGPHjvHiiy/SoUMHVq5cyZo1a2jatKnt1xbWrVvH2rVrKSkpYdy4cZw8eZJ3330XFxcXbr/9dp555hmOHTvGM888g2EYZUrk1Vdf5auvvqKkpIT777//gt+627hxI8uWLcPd3Z2WLVsyc+ZMpk6dyoEDB5g2bRozZ84sM//999/P5s2b6devHyUlJezevZtbb73VlvPnn38mLCzsgudhypQp3HfffXTt2pXo6GgOHTpEcXExjz32GPfddx+RkZG0b9+en376iby8PBYuXEiTJk0YP348eXl5FBQUEBUVxZ133lnh575x48a89957BAUF0bZtWz788MOLHjfbpEkTgoKC+PzzzwkJCanw+mu7oKCgMm+2U6dOZc6cObRt25Y1a9bwz3/+k5EjR+Li4sKAAQPIzc1l1qxZdterIQiptK+++orIyEjbf5s3b7a7TLNmzXjrrbeIjIxk9erV5Obm8t5775GYmEhcXFyZC8I3aNCA+Ph4AgICWLx4Me+++y7x8fEcPXqUlJQU3nnnHR544AGWL19u2zsF2LBhA6+88gorV6684CdqcnJyWLx4McuWLSM+Ph5vb29Wr17N9OnTadu27QXlC+eu6bF//35Onz7NV199ValCBFi9ejXXXHMNCQkJvPPOOyxYsMB2NbjAwEDeffddunfvzpYtWzh48CDHjx9n6dKlzJ8/3/ZzO/aUntiyZMkSzpw5w6RJk+jRowf/+Mc/yr1SWOPGjcnJyanUtkhZmZmZ/O///i+RkZGsXbuWY8eOsWHDBpo0acInn3zC1q1bee2118pc4e5itAcslXbXXXfZ9nLh3BjwxZxfAAEBAQBcf/31tl8Padu2rW0vrfQCRgCtWrUC4ODBg/z++++2j/T5+fkcOnSIn376yXaB7s6dOxMfHw+c+5mj2NhYjh8/Ts+ePctkOXToEG3btqV+/frAuQu+fPHFF/Tp0+eS23r33XezdetWvvzyS0aPHl1mu+3JzMykW7duANSvX582bdrYLrV48803256P48eP065dO4YNG8akSZM4e/bsBePapR9tz3f69Gk8PDz4448/+PXXX4mKiiIqKoqjR48yduxYOnTocNFcv/76q+3x5fK0atWKefPm0axZM9LS0sjOzqagoIB69erh6uqKl5cX7u7u5OfnX3I92gOWKuPh4UF2djYAhw8fLvOl1p9PQb7xxhvZt28fBQUFFBcXk5GRYbuv9BRuX19fbrjhBt5++22WL1/OI488QseOHWndujXffvstAN999x1w7nTVjz76iNjYWJYtW8b69es5fPiwbZ2+vr5kZmZy+vRpAL7++mtb0V/KwIED2bBhA9nZ2ZX+BYg2bdqwY8cOAPLy8vjxxx/x9fW96Lx79+4lPz+fN954gxdffPGCj6+NGzcmPz+fffv2Aee+PPzyyy+59dZbKSwsZMKECRw5cgQ4N3bdpEmTiw5BHDt2jK1bt9K7d+9KbYuUNWPGDCZPnkxERITtV1wGDhwIQFhYGGFhYQwcONDudT20ByxV5pZbbsHb25uQkBDatGlTbtkA+Pj4MH78eMLCwvDx8aFu3boXnWf48OFERkZSXFxM8+bNGTBgAOPHj2fixIl88MEHtsdwd3enYcOGBAcH07BhQ7p3717mUoY+Pj6MHTuWRx99FBcXF/z8/HjmmWdsbxjlad26NTk5OZd1Fa7Q0FCmTp1KeHg4VquVp59+msaNG1903pYtW/L666+zYcMG6tSpY7sYeymLxcLcuXOJjo7GxcWFoqIi7rnnHu666y4AYmJiePrpp3Fzc6O4uJg+ffrQo0cP1q1bx+bNm9m1axcuLi4YhsHcuXOr7UL8NZmvry+JiYnAudf68uXLL5jnYkNZl6JTkUVETKIhCBERk6iARURMogIWETGJClhExCQqYBERk6iARURMogIWETGJClhExCT/Dws4PYI8jU3WAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Create a distribution plot, displaying budget\n",
    "sns.displot(x='budget', #Set the x-axis\n",
    "            data=df, #Set the dataset\n",
    "            bins=20, #Set the number of bins\n",
    "            color='pink') #Make the bars pink\n",
    "\n",
    "plt.xticks(rotation=45, # Set x-axis tick labels rotation\n",
    "           horizontalalignment='right', # Set x-axis tick labels alignment to the right\n",
    "           fontweight='light', # Set x-axis tick font weight to light\n",
    "           fontsize='large') # Set x-axis tick labels size to large\n",
    "\n",
    "# Set x-axis label\n",
    "plt.xlabel('Hundreds of Millions USD')\n",
    "\n",
    "# Set y-axis label\n",
    "plt.ylabel('Number of occurrences')\n",
    "\n",
    "# Set plot title\n",
    "plt.title('Distribution of Movies Budget')\n",
    "\n",
    "# Display the plot\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0cd03910-fa79-4e4c-96cf-13dbb213cba1",
   "metadata": {},
   "source": [
    "##### The graph demonstrates the distribution of our movies' budget. Interestigly, the largest chunk of movies in this dataset have a budget of around 0.\n",
    "##### To answer the main question, I want to see if there any budgetless movies that made positive revenue."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a845791b-8760-424a-be17-e4c80a9ed724",
   "metadata": {},
   "source": [
    "### Checking the distribution of revenue for budgetless movies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "3c676263-b1f3-4c12-a40c-9ee2262db45b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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LuczWt3+81BPmEym8/aqWrRoQYcOHRg6dCghISGEhITQo0ePC85/++23V7js8PBwAAIDA2nXrp1zr6e60tPTWbt2LSaTCR8fH0aPHs3bb7/NH//4RwDuueceADp27EhRURG//vorZrPZOX9WVhYBAQF07twZgPbt29O1a1f++c9/cscdd1S67nOHIIqLi1mwYAGRkZGsXLmywnn279+Pl5eXc5s98MADzmWkpaVx33330ahRIwDGjBnjHGfdunUr+fn5fP755871VfZz/9dff2Xnzp388ssvLFu2zNmWk5NDr1698PT0ZOTIkfTq1YvQ0FCCg4M5derUBdsrMnjwYD744AMGDhxIamoqQ4cOZc+ePc51ffXVV84/hv7+/gwbNoz09HR+//vfc/PNN7N582Z69OjBF198waJFi8jLy3Mue8uWLezevZvhw4cDUFpayunTpwH4/e9/z6RJk+jTpw89e/bkscceu2B95w5BbNu2jenTp9O3b1/nHn1F29RkMjF8+HDef/99br31VlJSUpgxY0al2zQ4OBhvb2/69OkDwC233MLPP/9c4bYrs3XrVnbv3s26desAnH+M6zoFcC3YvXs3gYGB5do8PDxYtWoVu3fvZvv27SxevJjevXszY8aM8+Zv2LBhhcv28Pjfj5jS0lK8vLwwmUyce4uP4uJilzWWlpZiMpnKvT5z5ozzdVnYlk3z21uIlJSUlJu/bJpzl1EV3t7eWK1WRo0a5VxfRX35bQ1lPze9vLzKvXfuuHNpaSmzZs1yfsFtNtt5e/PnKi0txeFwkJSUhK+vLwAnT57EbDZjsVj44IMP+Oqrr/jiiy+YNm0ajz76KGPGjKmw/ULuuece58/xnTt3Mn/+fGcAl63/tzWVbdewsDBSU1M5ceIE/fv3x2KxlAvg0tLSckNGRUVF/PLLLwBERkYyfPhwMjIySElJ4Y033nAGWEV69+7NI488wtSpU9m4cSN+fn6VbtMRI0YwdOhQRo4cSX5+Pt27d6egoKDCbZqXl4e3t7fzc33uZ6qyz0JpaSnLli2jXbt2wNnhjt9+HusiDUHUsAMHDpCQkMC4cePKtefk5PDAAw/Qrl07/vSnPzF27FjnTzVPT88qB9f7778PwN69ezl48CCdO3emWbNm/Oc//8Fut1NcXMzf//535/QVLbtXr16sWrUKh8NBUVERycnJ3HXXXVXu52233ca3335LVlYWAP/5z3/YuXMn3bt3r/IyymzdutW5x9i0aVNnGB05coR//vOfwNkzQBwOB2lpaQBs2rTJGSx9+vThk08+IT8/H6BcqPTq1YvVq1dTVFREaWkpc+bMcR5IutC28fPz47bbbnP+nD116hTh4eFs2rSJLVu2MHbsWLp06cLkyZMZMmQIe/bsqbC9Ij4+PgwYMIAZM2bQr1+/cuOWfn5+dO7c2Xl2Rn5+Pqmpqc5/mwEDBrB3716Sk5PPG34o6++6deucZ5UsW7aMGTNmcObMGfr168fp06cJDw9n3rx57Nu3zzmcUZlx48ZhsVicB3Ar26YtWrQgODiYuXPnMmLECJfbtDJNmzZl7969OBwOCgoK2LJlS7l+vvXWW87P78SJE88746Yu0h7wZVY2XgZn907NZjNPPPEEd999d7npOnTowP3338/w4cNp2LAhDRo0cI7N9evXj/j4+Crtuf7www8MGTIEk8lEfHw8TZo0oWfPnvzud7/j/vvv5+qrr+aOO+5wjkHedtttvPzyy0yaNImIiAjncmJiYli4cCGDBg2iuLiY3r17M2HChCr3u1mzZixbtozY2FgKCwsxmUwsWbKENm3auDw17OOPPyYzMxOTyYTdbueGG24gLi4OgIiICJ566ilCQ0Np1aqV82wBb29vXn75ZebPn098fDxBQUHOoYQePXoQFhbGqFGjaNCgAe3bt3fuaf35z38mLi6OoUOHUlJSQlBQENHR0QCEhITw9NNPn1ffc889R2xsLIMGDaKoqIgHHniABx98kJKSEtLT03nggQdo2LAhjRs3JjY2luuuu+6C7ZUZPHgwVqvVeUDtt+tfsGABKSkpFBUVMWjQIIYNGwacDe+BAwfy+eefX3CYY+TIkRw5coSwsDBMJhPXXXcdTz/9NF5eXsyaNYunnnrK+atp8eLF+Pj4VFpn2bafM2cO48ePZ8SIEZVu07Iapk6dyooVK1xu08o+Kw8++CDbtm3j3nvvpUWLFnTv3t25Rzx79mwWLVrk/PzeddddjB8/3mVfjGbS7Silvtm9ezf/+te/+MMf/gDAm2++ya5du3jhhReMLUzkNxTAUu8UFBQwa9Ysvv32W+deX2xsLC1atDC6NJFyFMAiIgbRQTgREYMogEVEDKIAFhExSL0J4P/85z/Vnue77767/IXUInevH9y/D6rfWO5ef70J4OpecQU4L8l0V+5eP7h/H1S/sdy9/noTwCIi7kYBLCJiEAWwiIhBFMAiIgZRAIuIGEQBLCJikBoL4BMnTtCnTx9yc3P5/vvvCQ8Px2q1Mm/ePOezrpKTkxk2bBhhYWHOe3sWFhYyefJkrFYrjz32WLWf+isi4i5qJICLi4uZO3eu8zE8S5YsYdq0aaxZswaHw8GmTZs4duwYiYmJJCUlsXLlSuLj4ykqKmLt2rUEBgayZs0ahgwZQkJCQk2UKCJiuBoJ4Li4OEaPHs0111wDnH1aQ9mTEUJCQvj888/JysqiS5cu+Pj44O/vT0BAADk5OWRmZtK7d2/ntNu3b6+JEkVEDHfZn4iRkpJCs2bN6N27N6+99hpw9tldZc9nslgs5OfnU1BQgL+/v3M+i8VCQUFBufayaavCbreTnZ1drVoLCwurPU9d4u71g/v3QfUby13qDwoKumD7ZQ/g9evXYzKZ2L59O9nZ2URFRZUbx7XZbDRq1Ag/P79yjwu32Wz4+/uXay+btirMZnOFnaxIdnZ2teepS9y9fnD/Pqh+Y7l7/Zd9CGL16tWsWrWKxMREgoKCiIuLIyQkhB07dgBnH3/erVs3goODyczMxG63k5+fT25uLoGBgXTt2tX5oMX09PRKH8kuIuLOauWhnFFRUc4npbZt25bQ0FA8PT2JiIjAarXicDiIjIzEbDYTHh5OVFQU4eHheHt7s3Tp0hqrq01Am4uar6SoBE8fT9cTiohUokYDODEx0fn/F3pEdFhY2HmP0vb19XU+7rqmNbA04O2+b1d7voe3PFwD1YjIlUYXYoiIGEQBLCJiEAWwiIhBFMAiIgZRAIuIGEQBLCJiEAWwiIhBFMAiIgZRAIuIGEQBLCJiEAWwiIhBFMAiIgZRAIuIGEQBLCJiEAWwiIhBFMAiIgZRAIuIGEQBLCJiEAWwiIhBFMAiIgZRAIuIGEQBLCJiEAWwiIhBFMAiIgZRAIuIGEQBLCJiEAWwiIhBFMAiIgZRAIuIGMSrJhZaUlJCTEwMBw4cwNPTkyVLlpCfn8+ECRO48cYbAQgPD2fgwIEkJyeTlJSEl5cXEydOpG/fvhQWFjJ9+nROnDiBxWIhLi6OZs2a1USpIiKGqZEA3rJlCwBJSUns2LGDJUuW0K9fPx555BHGjRvnnO7YsWMkJiayfv167HY7VquVnj17snbtWgIDA5k8eTIbN24kISGBmJiYmihVRMQwNRLA/fv35+677wbgv//9L82bN2fPnj0cOHCATZs20bp1a2bNmkVWVhZdunTBx8cHHx8fAgICyMnJITMzk/HjxwMQEhJCQkJCTZQpImKoGglgAC8vL6Kiovj0009Zvnw5R44cYeTIkXTq1IkVK1bw8ssv06FDB/z9/Z3zWCwWCgoKKCgocLZbLBby8/Ndrs9ut5OdnV2tGoOCgrD9aqtex/5PdddVEwoLC+tEHZfC3fug+o3lLvUHBQVdsL3GAhggLi6Op556irCwMJKSkmjRogUAAwYMIDY2lm7dumGz/S8AbTYb/v7++Pn5OdttNhuNGjVyuS6z2VxhJytjaWip9jxQ8QatTdnZ2XWijkvh7n1Q/cZy9/pr5CyI1NRUXn31VQB8fX0xmUxMmjSJrKwsALZv307Hjh0JDg4mMzMTu91Ofn4+ubm5BAYG0rVrV9LS0gBIT0/n9ttvr4kyRUQMVSN7wPfeey8zZ85kzJgxnDlzhlmzZnHdddcRGxuLt7c3zZs3JzY2Fj8/PyIiIrBarTgcDiIjIzGbzYSHhxMVFUV4eDje3t4sXbq0JsoUETFUjQRww4YNWbZs2XntSUlJ57WFhYURFhZWrs3X15fly5fXRGkiInWGLsQQETGIAlhExCAKYBERgyiARUQMogAWETGIAlhExCAKYBERgyiARUQMogAWETGIAlhExCAKYBERgyiARUQMogAWETGIAlhExCAKYBERgyiARUQMogAWETGIAlhExCAKYBERgyiARUQMogAWETGIAlhExCAKYBERgyiARUQMogAWETGIAlhExCAKYBERgyiARUQMogAWETGIAlhExCBeNbHQkpISYmJiOHDgAJ6enixZsgSHw0F0dDQmk4n27dszb948PDw8SE5OJikpCS8vLyZOnEjfvn0pLCxk+vTpnDhxAovFQlxcHM2aNauJUkVEDFMje8BbtmwBICkpiSlTprBkyRKWLFnCtGnTWLNmDQ6Hg02bNnHs2DESExNJSkpi5cqVxMfHU1RUxNq1awkMDGTNmjUMGTKEhISEmihTRMRQNbIH3L9/f+6++24A/vvf/9K8eXO2bt1K9+7dAQgJCSEjIwMPDw+6dOmCj48PPj4+BAQEkJOTQ2ZmJuPHj3dOW5UAttvtZGdnV6vOoKAgbL/aqte5/1PdddWEwsLCOlHHpXD3Pqh+Y7lL/UFBQRdsr5EABvDy8iIqKopPP/2U5cuXs2XLFkwmEwAWi4X8/HwKCgrw9/d3zmOxWCgoKCjXXjatK2azucJOVsbS0FLteaDiDVqbsrOz60Qdl8Ld+6D6jeXu9dfoQbi4uDj+/ve/M2fOHOx2u7PdZrPRqFEj/Pz8sNls5dr9/f3LtZdNKyJS39RIAKempvLqq68C4Ovri8lkolOnTuzYsQOA9PR0unXrRnBwMJmZmdjtdvLz88nNzSUwMJCuXbuSlpbmnPb222+viTJFRAxVI0MQ9957LzNnzmTMmDGcOXOGWbNm0a5dO+bMmUN8fDxt27YlNDQUT09PIiIisFqtOBwOIiMjMZvNhIeHExUVRXh4ON7e3ixdurQmyhQRMVSNBHDDhg1ZtmzZee2rVq06ry0sLIywsLBybb6+vixfvrwmShMRqTN0IYaIiEEUwCIiBlEAi4gYRAEsImIQBbCIiEEUwCIiBlEAi4gYRAEsImIQBbCIiEEUwCIiBlEAi4gYRAEsImIQBbCIiEFcBvDOnTtJT08nLS2N/v378+GHH9ZGXSIi9Z7LAH722We58cYbeeedd1i7di1JSUm1UZeISL3nMoDNZjNXXXUVXl5eXH311RQVFdVGXSIi9Z7LAPbz8+ORRx7h/vvvZ/Xq1Vx33XW1UZeISL3n8okYy5Yt4+DBg9x0003s37+fkSNH1kZdIiL1nssAzsvL45VXXiEvL4/Q0FBOnz5N586da6M2EZF6zeUQxJw5cxg+fDhFRUV069aNRYsW1UZdIiL1nssAttvt9OjRA5PJRNu2bTGbzbVRl4hIvecygH18fNi2bRulpaV8/fXX+Pj41EZdIiL1nssAjo2NJSUlhby8PN544w3mz59fC2WJiNR/VToPeMSIEWzcuJHu3bvTuHHj2qhLRKTecxnATzzxBPn5+QA0btyY6dOn13hRIiJXApcBfPr0ae677z4ABg0axOnTp2u8KBGRK4HLAPb29iYjI4OCggK2b9+Oh4duoCYicjm4TNOFCxeyevVqRo4cyZo1a1iwYEFt1CUiUu+5vBKudevWJCQk1EYtIiJXFJcB/Morr/DXv/6VBg0aONv+8Y9/VDh9cXExs2bN4scff6SoqIiJEydy7bXXMmHCBG688UYAwsPDGThwIMnJySQlJeHl5cXEiRPp27cvhYWFTJ8+nRMnTmCxWIiLi6NZs2aX3lMRkTrGZQD/7W9/Y9u2bfj6+lZpgRs2bKBJkyY8++yz5OXlMXToUB5//HEeeeQRxo0b55zu2LFjJCYmsn79eux2O1arlZ49e7J27VoCAwOZPHkyGzduJCEhgZiYmIvvoYhIHeVyDLhly5bl9n5due+++5g6darztaenJ3v27GHr1q2MGTOGWbNmUVBQQFZWFl26dMHHxwd/f38CAgLIyckhMzOT3r17AxASEsL27dsvolsiInWfyz3g4uJiBg0aRGBgICaTCYClS5dWOL3FYgGgoKCAKVOmMG3aNIqKihg5ciSdOnVixYoVvPzyy3To0AF/f/9y8xUUFFBQUOBst1gsznOQXbHb7WRnZ1dp2jJBQUHYfrVVa54y1V1XTSgsLKwTdVwKd++D6jeWu9QfFBR0wXaXAfzYY49Ve2WHDx/m8ccfx2q1MmjQIE6dOkWjRo0AGDBgALGxsXTr1g2b7X/hZ7PZ8Pf3x8/Pz9lus9mc87liNpsr7GRlLA0t1Z4HKt6gtSk7O7tO1HEp3L0Pqt9Y7l6/yyGIW265hYyMDFJTU/n5559p0aJFpdMfP36ccePGMX36dEaMGAHAo48+SlZWFgDbt2+nY8eOBAcHk5mZid1uJz8/n9zcXAIDA+natStpaWkApKenc/vtt19qH0VE6iSXe8CzZs0iJCSEnTt30rx5c2bPns2qVasqnP6VV17h1KlTJCQkOE9fi46OZvHixXh7e9O8eXNiY2Px8/MjIiICq9WKw+EgMjISs9lMeHg4UVFRhIeH4+3tXelwh4iIO3MZwD///DMjRoxgw4YNdO3aFYfDUen0MTExFzxr4UJPUw4LCyMsLKxcm6+vL8uXL3dVloiI26vSdcW5ubkA/PTTT7oUWUTkMnGZpjExMcyaNYt///vfTJkyhejo6NqoS0Sk3nM5BLFt2zbefffd2qhFROSK4nIPOC0tjZKSktqoRUTkilKlx9L37t2bVq1aYTKZMJlMFzygJiIi1eMygF944YVqXYosIiJV4zKAY2JiWLt2bW3UIiJyRXEZwA0bNmTx4sW0adPGeQraqFGjarwwEZH6zmUAd+nSBYATJ07UeDEiIlcSlwE8bNiw2qhDROSK4zKAIyMjMZlMlJaWcujQIVq3bq0xYRGRy8BlAJ97EcapU6eYO3dujRYkInKlqNaNHfz9/Tl48GBN1SIickVxuQc8atQo55MwTpw4QY8ePWq8KBGRK4HLAI6Pj8fhcDivgrv++utroy4RkXrP5RBERkYG77zzDi1btiQmJobU1NRaKEtEpP5zGcBr167lySefBODVV1/VGRAiIpeJywD28PDAbDYD4O3t7RwPFhGRS+NyDPiee+7BarUSHBzM3r176devX23UJSJS77kM4D//+c/07duXAwcOMHDgQIKDg2ujLhGRes/lEERycjLvv/8+AwcO5IUXXtBBOBGRy0QH4UREDKKDcCIiBtFBOBERg1TrINyQIUPo0KFDbdQlIlLvuQzgn376iRUrVvDNN9/Qpk0bZs6cSatWrWqjNhGRes3lGHBMTAyDBw8mKSmJoUOHMnv27NqoS0Sk3nMZwHa7nXvuuYdGjRrRv39/SkpKaqMuEZF6z2UAl5SUsG/fPgDnf0VE5NK5HAOeM2cOs2bN4tixY1xzzTUsXLiw0umLi4uZNWsWP/74I0VFRUycOJGbbrqJ6OhoTCYT7du3Z968eXh4eJCcnExSUhJeXl5MnDiRvn37UlhYyPTp0zlx4gQWi4W4uDiaNWt22TosIlJXuAzgoKAg1q9fX+UFbtiwgSZNmvDss8+Sl5fH0KFD6dChA9OmTeOOO+5g7ty5bNq0idtuu43ExETWr1+P3W7HarXSs2dP1q5dS2BgIJMnT2bjxo0kJCQQExNzSZ0UEamLXAZwdd13332EhoY6X3t6erJ37166d+8OQEhICBkZGXh4eNClSxd8fHzw8fEhICCAnJwcMjMzGT9+vHPahISEy12iiEidUGEA5+fn4+/vX+0FWiwWAAoKCpgyZQrTpk0jLi7OeQWdxWIhPz+fgoKCcsu3WCwUFBSUay+btirsdjvZ2dnVqjUoKAjbr7ZqzVOmuuuqCYWFhXWijkvh7n1Q/cZyl/qDgoIu2F5hAE+YMIHVq1czb948/vKXv1RrZYcPH+bxxx/HarUyaNAgnn32Wed7NpuNRo0a4efnh81mK9fu7+9frr1s2qowm80VdrIyloaWas8DFW/Q2pSdnV0n6rgU7t4H1W8sd6+/wgBu0KABw4cP5/vvv3ee/VD2bLikpKQKF3j8+HHGjRvH3LlznQ/wvOWWW9ixYwd33HEH6enp3HnnnQQHB/PCCy9gt9spKioiNzeXwMBAunbtSlpaGsHBwaSnp3P77bdf5i6LiNQNFQbw66+/ztGjR5k7dy7z58/H4XBUaYGvvPIKp06dIiEhwTl+O3v2bBYuXEh8fDxt27YlNDQUT09PIiIisFqtOBwOIiMjMZvNhIeHExUVRXh4ON7e3ixduvTy9FREpI4xOVwk65kzZ3j33Xf55ptvuPHGGwkPD8fHx6e26quyi/0p8nbft6s9z8NbHq72PDXB3X9+gfv3QfUby93rd3khxty5czl48CA9e/bkxx9/1ClhIiKXicvT0L7//ntWr14NQP/+/Rk9enSNFyUiciWo0r0gTp8+DZw95UP3ghARuTxc7gH/4Q9/YPDgwbRv355vvvmGKVOm1EZdIiL1nssAfvDBBwkJCeGHH36gVatWNG3atDbqEhGp96p0KXKTJk1o0qRJDZciInJlcTkGLCIiNcNlAK9cubI26hARueK4DOC0tDSd+SAiUgNcjgHn5eXRu3dvWrVqhclkcnkvCBERqRqXAfzKK6/URh0iIlcclwHs5eXlfLpFaGgoN998My1btqyN2kRE6jWXY8Bz5sxh+PDhFBUV0a1bNxYtWlQbdYmI1HtVuhS5R48emEwm2rZti9lsro26RETqPZcB7OPjw7Zt2ygtLeXrr7+uk7eiFBFxRy4DODY2lpSUFPLy8njjjTeYP39+LZQlIlL/uTwId+211/KnP/2J7777jvbt23PDDTfURl0iIvWeywBOSEhg27Zt3Hrrrbz11lvcd999jB07thZKExGp31wGcHp6OmvWrMHDw4MzZ85gtVoVwCIil4HLMeBmzZo5b8heXFxMs2bNarwoEZErQYV7wKNGjcJkMnHixAnnBRi5ubm6LaWIyGVSYQDHx8fXZh0iIlecCgO47HLjrKwsNm7ciN1ud76nU9FERC6dy4NwUVFRPPbYYzRq1Kg26hERuWK4DODWrVszbNiw2qhFROSK4jKAQ0NDiYyMpF27ds62SZMm1WhRIiJXApcBvGbNGgYMGKAhCBGRy8xlADdu3Jg//vGPtVGLiMgVxWUAN23alLlz53LLLbdgMpmAs+cIi4jIpXF5JVzr1q255pprOH78OMeOHePYsWNVWvCuXbuIiIgAYO/evfTu3ZuIiAgiIiL4+OOPAUhOTmbYsGGEhYWxZcsWAAoLC5k8eTJWq5XHHnuMkydPXmzfRETqNJd7wBdzBsTrr7/Ohg0b8PX1BeDf//43jzzyCOPGjXNOc+zYMRITE1m/fj12ux2r1UrPnj1Zu3YtgYGBTJ48mY0bN5KQkEBMTEy1axARqetcBnBkZCQmk4nS0lIOHTpE69atWbt2baXzBAQE8OKLLzJjxgwA9uzZw4EDB9i0aROtW7dm1qxZZGVl0aVLF3x8fPDx8SEgIICcnBwyMzMZP348ACEhISQkJFyGboqI1D0uA/jdd991/v+pU6eYO3euy4WGhoZy6NAh5+vg4GBGjhxJp06dWLFiBS+//DIdOnTA39/fOY3FYqGgoICCggJnu8ViIT8/v0odsdvtZGdnV2naMkFBQdh+tVVrnjLVXVdNKCwsrBN1XAp374PqN5a71B8UFHTBdpcBfC5/f38OHjxY7ZWfexrbgAEDiI2NpVu3bths/ws/m82Gv78/fn5+znabzVbl09/MZnOFnayMpaGl2vNAxRu0NmVnZ9eJOi6Fu/dB9RvL3et3eRBu1KhRjB49mlGjRnHvvffSqVOnaq/k0UcfJSsrC4Dt27fTsWNHgoODyczMxG63k5+fT25uLoGBgXTt2pW0tDTg7L2Ib7/99mqvT0TEHbjcAz73rmhms5nmzZtXeyXz588nNjYWb29vmjdvTmxsLH5+fkRERGC1WnE4HERGRmI2mwkPDycqKorw8HC8vb1ZunRptdcnIuIOKgzg1NTUCmcaMmSIywW3atWK5ORkADp27EhSUtJ504SFhREWFlauzdfXl+XLl7tcvoiIu6swgHNzc8u9djgcpKSk0KBBgyoFsIiIVK7CAH7yySed///9998THR3N3XffzaxZs2qlMBGR+s7lGPDq1at5++23mTlzJn379q2NmkRErggVBvCRI0eYOXMmjRs35r333qNx48a1WZeISL1XYQA/8MADeHt7c+edd7JgwYJy7+nMBBGRS1dhAL/88su1WYeIyBWnwgDu3r17bdYhInLFcXklnIiI1AwFsIiIQRTAIiIGUQCLiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASwiYhAFsIiIQRTAIiIGqbEA3rVrFxEREQB8//33hIeHY7VamTdvHqWlpQAkJyczbNgwwsLC2LJlCwCFhYVMnjwZq9XKY489xsmTJ2uqRBERQ9VIAL/++uvExMRgt9sBWLJkCdOmTWPNmjU4HA42bdrEsWPHSExMJCkpiZUrVxIfH09RURFr164lMDCQNWvWMGTIEBISEmqiRBERw3nVxEIDAgJ48cUXmTFjBgB79+6le/fuAISEhJCRkYGHhwddunTBx8cHHx8fAgICyMnJITMzk/HjxzunrWoA2+12srOzq1VnUFAQtl9t1ZqnTHXXVRMKCwvrRB2Xwt37oPqN5S71BwUFXbC9RgI4NDSUQ4cOOV87HA5MJhMAFouF/Px8CgoK8Pf3d05jsVgoKCgo1142bVWYzeYKO1kZS0NLteeBijdobcrOzq4TdVwKd++D6jeWu9dfKwfhPDz+txqbzUajRo3w8/PDZrOVa/f39y/XXjatiEh9VCsBfMstt7Bjxw4A0tPT6datG8HBwWRmZmK328nPzyc3N5fAwEC6du1KWlqac9rbb7+9NkoUEal1NTIE8VtRUVHMmTOH+Ph42rZtS2hoKJ6enkRERGC1WnE4HERGRmI2mwkPDycqKorw8HC8vb1ZunRpbZQoIlLraiyAW7VqRXJyMgBt2rRh1apV500TFhZGWFhYuTZfX1+WL19eU2WJiNQZuhBDRMQgCmAREYMogEVEDKIAFhExiAJYRMQgCmAREYMogEVEDKIAFhExiAJYRMQgCmAREYMogEVEDKIAFhExiAJYRMQgCmAREYMogEVEDKIAFhExiAJYRMQgCmAREYMogEVEDKIAFhExiAJYRMQgCmAREYMogEVEDKIAFhExiAJYRMQgCmAREYMogEVEDKIAFhExiAJYRMQgXrW5siFDhuDv7w9Aq1atmDBhAtHR0ZhMJtq3b8+8efPw8PAgOTmZpKQkvLy8mDhxIn379q3NMkVEakWtBbDdbgcgMTHR2TZhwgSmTZvGHXfcwdy5c9m0aRO33XYbiYmJrF+/HrvdjtVqpWfPnvj4+NRWqSIitaLWAjgnJ4fTp08zbtw4zpw5wxNPPMHevXvp3r07ACEhIWRkZODh4UGXLl3w8fHBx8eHgIAAcnJyCA4Orq1SRURqRa0FcIMGDXj00UcZOXIk3333HY899hgOhwOTyQSAxWIhPz+fgoIC5zBFWXtBQYHL5dvtdrKzs6tVU1BQELZfbdXryP+p7rpqQmFhYZ2o41K4ex9Uv7Hcpf6goKALttdaALdp04bWrVtjMplo06YNTZo0Ye/evc73bTYbjRo1ws/PD5vNVq793ECuiNlsrrCTlbE0tFR7Hqh4g9am7OzsOlHHpXD3Pqh+Y7l7/bV2FsS6det4+umnAThy5AgFBQX07NmTHTt2AJCenk63bt0IDg4mMzMTu91Ofn4+ubm5BAYG1laZVVJSVFIr84hI/VZre8AjRoxg5syZhIeHYzKZWLx4MU2bNmXOnDnEx8fTtm1bQkND8fT0JCIiAqvVisPhIDIyErPZXFtlVomnjydv9327WvM8vOXhGqpGRNxVrQWwj48PS5cuPa991apV57WFhYURFhZWG2WJiBhGF2KIiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASwiYhAFcC252Idy6mGeIvVXrT0T7kp3MQ/yBD3MU6Q+0x6wiIhBFMAiIgZRAIuIGEQBLCJiEAWwiIhBFMAiIgZRANdxlZ0HHBQUVO15RKTu0HnAdVxl5w/bfrVhaWg5r13nDou4B+0Bi4gYpE4GcGlpKXPnzmXUqFFERETw/fffG12SW7mYIYgz9jO1ti4ROatODkF89tlnFBUV8e677/L111/z9NNPs2LFCqPLchsXc9nzw1sevqhLpR/6+0PVnqekqARPH89qzydS39TJAM7MzKR3794A3HbbbezZs8fgiqQiFxP254Z2RQcSL0TBLfWNyeFwOIwu4rdmz57NvffeS58+fQC4++67+eyzz/Dyqvjvxddff43ZbK6tEkVEqszLy4v27duf325ALS75+flhs9mcr0tLSysNXzi7pywi4k7q5EG4rl27kp6eDpzdsw0MDDS4IhGRy69ODkGUlpYyf/589u/fj8PhYPHixbRr187oskRELqs6GcAiIleCOjkEISJyJVAAi4gYRAEsImKQehnA5w5ra4hbLpY7f3b0HXAP9TKACwsLnf9vMpkoLS01sJqLk5GRwapVqwB9gYxy6tQpo0u4aPoOuId6F8Dbtm0jMjKShQsXsnjxYgA8PDzc6h8wLS2NpUuXEhwcDJz9AoF7fQgPHjzIyZMnOXz4sNGlXJS0tDRmzZrFqVOnKClxrxsO6TvgPupVAGdmZjJ37lyGDBlCz549ycrKYuTIkZSUlLjNXsC2bdtYtGgRy5cvB2DdunW8/PLL2O1254ewrtu8eTNTp04lNjaWSZMm8d577xldUrVs3bqV5cuXM2nSJBo1auRWAazvgHupV+cBb9iwgcOHD/OnP/3J2Wa1WikpKeHdd98Fzv4Frav/iIWFhSxYsIDWrVsTFBTEs88+y4ABA0hLS8PhcPDmm2/SuHHjOt2HQ4cOMX78eJ555hluuOEGvvrqK6ZMmcL06dMZO3as0eW5tGvXLqZOncobb7zBoUOH+H//7/9x6NAhQkNDGTNmjNHluaTvgHupV3vApaWlrF+/niNHjjjb1qxZg6enJ8888wxAnf5Ha9CgAf3792ft2rWsXLmSt956iylTprB+/XqaNWvG3Llzgbrdh8LCQlq2bElwcDBNmzblnnvu4a233uK5554jNTXV6PJcOnDgALfccgv79u3j1VdfpV+/fgwaNIjly5fz2muvGV2eS/oOuJd6FcADBgzgrrvu4q233uLEiRPO9oceeqjO/4ws+yHSr18/xo0bR0FBAZ6ens6DKVOnTqVBgwZGllglAQEB5OXl8c477zjbfve737F06VKSkpI4evRonR7HGzJkCCaTiTlz5hAdHU3//v0ZOXIkr732GqtXr+bAgQNGl3hBZdtU3wH3Um8CuLS0FIvFQkhICL/88gt//etfnXsBx44d49ChQxQVFdXZL7/JZKK4uBg4+2VZunQpTZo04cyZs0+qyMrK4sSJE9jt9jrbh9LSUnx8fIiIiGDXrl18+OGHzve6dOnC1VdfTYMGDers3kvZ+Gh8fDx33XUXHh7/+3p07tyZrl270rBhQ6PKq1TZ+K67fweKiooA9/0OVFedvB1lVXz11Vd8++23+Pr60qNHD5o1awac/evpcDjIyMggPDyc0NBQNm3aREJCAj4+PgZXXd6F+lBaWoqHhwetWrUiPT2dpUuXctddd7F582ZeeumlOnfP4+LiYry9vQGcgdWrVy+OHj3K1q1bOXHiBGPHjuXLL7/kp59+cv6RqSt+W39paSlms5m4uDh8fX3ZtWsXnTt35uOPP+b7778vF8pGy8jIICMjg+PHjzN58mRuuOEGwL2+Axfqg7t9By6FWx6E27RpE8899xy9evXi+PHj7Ny5kzfffLPcDY+LiorYsmULDRo0oE2bNgQEBBhY8fkq68OZM2fw8vLil19+4aOPPsLf35/OnTvTunVro8suZ//+/Wzfvp3BgwfTpEkT516JyWTi2LFjbN++nbfeeourrrqKn376ieeee46bb77Z4Kr/57f1n8vhcJCbm8ucOXPw9PTk559/ZunSpXWm/q1bt7Js2TKsVitZWVmkpaWRmprq3BGBuv8dqEof6vp34JI53ExRUZEjMjLSsXPnTmfbM8884+jSpYsjNzfXwMqqzt37UFpa6nA4HI533nnH0b9/f8eqVascJ06ccL535swZ57TFxcWOQ4cOOfLy8owo9YIqq//c90tKShxHjx517Nu3r9z7Rvv1118df/7znx27du1yto0ePdr5uqz+usxVH64Udef3VBU5HA4OHz7Md99952ybPn064eHhjB8/nvz8fLZv384TTzzhnL6ucfc+lI3hHj58GLPZzJ49e/joo484efIkJpPJ+TN93759eHl50bJly/P2MI1UWf3nys3N5eqrryYwMLDcXpnRHA4HBw8e5JdffnG2nT592jneazKZ2L59O1OmTHFOX9e46gNQ5/twObhdAPv4+DBq1Cg2b95MVlaWs3369Ol06tSJHTt2cOONNzJt2jSgbp6uUh/6ANC4cWNmzJhB586d+fLLL8uF8Hfffcfs2bPJy8urs18eV/XPnDmTkydP1rn6GzZsyNSpU2ncuDFFRUWcOXMGHx8f5xDDZ599RsOGDYmKigLq5uenPvThcnDLg3B9+vRh3759ziusyi5XbNCgATabjeuuu87I8qqkPvRhwIABtGjRgpCQEEpLS9mxYweenp7cf//93Hjjjbzxxhs0atTI6DIr5M719+nTB09PTzw8PDh+/DjHjx/n+uuv529/+xvPP/88r732Gi1btjS6zErVhz5cKrcM4KZNm/Lwww/z9ttvk5CQQJ8+ffD19SUnJ4fHH3/c6PKqpD70oW3bts7/t1qteHp68sknn+Dl5cXIkSPrbHiVcef6y87cADCbzQQGBvLee+/x0UcfkZCQwI033mhccVVUH/pwqdzyLIgyv/zyC//4xz/46KOPuOqqq3jooYfo0KGD0WVVS33og+Ocy0LXr19Pr169aNGihcFVVZ2713/y5EnuuusuWrduzYoVK8r9YXEX9aEPF8OtA7hM2Y1G6tI5mtXl7n1wuPm1+e5c/5kzZ1i0aBEPPfSQ2z68tj704WLUiwAWudKde0GJu6oPfaguBbCIiEHc8/euiEg9oAAWETGIAlhEpAp27dpFREREpdMsXLiQYcOGOe8I6IpbngcsV54dO3Ywbdo0brrpJhwOh/Ooebt27Vi0aBGPPPII69evp3nz5rRt25akpCSef/55Jk2axEsvvXRZa3nxxRdp3rw54eHhzrawsDDi4+O5/vrriYuLY//+/Xh4eODt7c3s2bO54YYbiIiI4PTp0/j6+lJcXEyrVq2YPXs2TZs2vaz1yeX3+uuvs2HDBnx9fSucZsuWLRw4cIB169bx888/M378eFJSUipdrvaAxW3ceeedJCYmsmrVKiZNmuR8wsPs2bO5/vrrLzjP5Q5fV7Zt28bRo0d58803WblyJSNGjHA+GBMgLi6OxMREkpKSCAkJcT7hQeq2gIAAXnzxRefrffv2ERERQUREBJMnTyY/P59vvvmG3r174+HhQbNmzfD09OTYsWOVLlcBLG7p1KlTzstUIyIiyM3NveB0PXv2BODf//434eHhPPTQQzz66KP897//5dChQ4waNYqpU6cybNgw5s2bB5x9sGVYWBhWq5UJEyZQUFBQ5bquvfZa9uzZw8cff8zJkye55557WLZs2QWnffDBB9m7dy92u706XRcDhIaG4uX1vwGDOXPmMG/ePBITEwkJCeGvf/0rQUFBbNu2jeLiYn744Qe++eYbTp8+XelyNQQhbuOLL74gIiKCoqIi5zPbqiomJoZFixYRFBTEZ599xtNPP82MGTP47rvvWLlyJb6+vvTv359jx47x2WefMWDAAB599FE2b97MqVOn8PPzc7kOk8nEzTffTGxsLMnJySxcuJBrr72W6OhounfvfsF5GjVqxKlTp7j66qur3BcxXm5uLn/5y1+As+cvt2nThl69erF7924efvhhOnToQMeOHV3eBVABLG7jzjvv5Pnnnwfg22+/ZfTo0aSnp1dp3qNHjxIUFAT87xl1cPanZVm4Xn311djtdiZMmMArr7zCww8/TIsWLZw3SipjNpudj84p8+uvv9KgQQNycnJo06YN8fHxzqdSTJs2jYyMjPNqcjgcHD9+nKuuuqp6G0IM16ZNG+Li4rj++uvJzMzk2LFjHDhwgKuuuoo1a9Zw+PBhZsyY4fJ+IhqCELfUvHnzak1/zTXXkJOTA8DOnTudN3q50OXHH374IUOHDiUxMZH27duTnJxc7v2OHTuyefNm57PKDh48SFFREVdddRXbt28nPj7eeWl5+/bt8fX1veB61q1bx5133um2l59fyebPn09UVBRWq9X5pJTrr7+ebdu2ERYWxowZM6o0vq89YHEbZUMQHh4e2Gw2oqOjq/yU3IULFxIbG4vD4cDT07PcgbHfuvXWW4mOjqZhw4Z4e3uzYMGCcu/37NmTr776imHDhuHn54fD4SAuLg44Ox4dFxfHkCFD8PPzw8PDw3mwECAqKsp5JL1FixbOcWep+1q1auX8Y9ypUycSExPPm+bcA3VVoUuRRUQMot8+IiIGUQCLiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASwiYhAFsIiIQf4/4UGuKNZJpBsAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Create a distribution plot, displaying revenue\n",
    "sns.displot(x='revenue', #Set the x-axis\n",
    "            data=df, #Set the dataset\n",
    "            bins=20, #Set the number of bins\n",
    "            color='purple') #Make the bars purple\n",
    "\n",
    "plt.xticks(rotation=45, # Set x-axis tick labels rotation\n",
    "           horizontalalignment='right', # Set x-axis tick labels alignment to the right\n",
    "           fontweight='light', # Set x-axis tick font weight to light\n",
    "           fontsize='large') # Set x-axis tick labels size to large\n",
    "\n",
    "# Set x-axis label\n",
    "plt.xlabel('Billions USD')\n",
    "\n",
    "# Set y-axis label\n",
    "plt.ylabel('Number of occurrences')\n",
    "\n",
    "# Set plot title\n",
    "plt.title('Distribution of Budgetless Movies Revenue')\n",
    "\n",
    "# Display the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a29ace3a-7e98-42d0-aa9b-a2e98cfb8444",
   "metadata": {},
   "source": [
    "##### From the plot we can notice that indeed there are some movies, which made positive revenue, regardless of their budget being 0. Since it seems suspicious, I investigate further.  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "162bc61c-096a-4ef1-99b3-d5ee4c8ad80c",
   "metadata": {},
   "source": [
    "### Finding a movie, which made the most revenue with 0 budget"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 481,
   "id": "3a9c0bf8-646a-408e-b63a-666c892673e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "zero_budget_movies = df[df.budget == 0] #Create a dataset only with movies, which budget was set to 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 483,
   "id": "e4687dd4-266e-4017-a6e0-0347bea65ab1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>budget</th>\n",
       "      <th>genres</th>\n",
       "      <th>homepage</th>\n",
       "      <th>id</th>\n",
       "      <th>keywords</th>\n",
       "      <th>original_language</th>\n",
       "      <th>original_title</th>\n",
       "      <th>overview</th>\n",
       "      <th>popularity</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>production_countries</th>\n",
       "      <th>release_date</th>\n",
       "      <th>revenue</th>\n",
       "      <th>runtime</th>\n",
       "      <th>spoken_languages</th>\n",
       "      <th>status</th>\n",
       "      <th>tagline</th>\n",
       "      <th>title</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>extracted_genres</th>\n",
       "      <th>extracted_production_companies</th>\n",
       "      <th>extracted_production_countries</th>\n",
       "      <th>extracted_keywords</th>\n",
       "      <th>extracted_spoken_languages</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>691</th>\n",
       "      <td>0</td>\n",
       "      <td>[{\"id\": 28, \"name\": \"Action\"}, {\"id\": 12, \"nam...</td>\n",
       "      <td>http://video.movies.go.com/wildhogs/</td>\n",
       "      <td>11199</td>\n",
       "      <td>[{\"id\": 1599, \"name\": \"midlife crisis\"}, {\"id\"...</td>\n",
       "      <td>en</td>\n",
       "      <td>Wild Hogs</td>\n",
       "      <td>Restless and ready for adventure, four suburba...</td>\n",
       "      <td>31.719463</td>\n",
       "      <td>[{\"name\": \"Wild Hogs Productions\", \"id\": 6354}...</td>\n",
       "      <td>[{\"iso_3166_1\": \"US\", \"name\": \"United States o...</td>\n",
       "      <td>2007-03-02</td>\n",
       "      <td>253625427</td>\n",
       "      <td>100.0</td>\n",
       "      <td>[{\"iso_639_1\": \"en\", \"name\": \"English\"}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>A lot can happen on the road to nowhere.</td>\n",
       "      <td>Wild Hogs</td>\n",
       "      <td>5.6</td>\n",
       "      <td>648</td>\n",
       "      <td>[Action, Adventure, Comedy]</td>\n",
       "      <td>[Wild Hogs Productions, Touchstone Pictures]</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[midlife crisis, road trip, politically incorr...</td>\n",
       "      <td>[English]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     budget                                             genres  \\\n",
       "691       0  [{\"id\": 28, \"name\": \"Action\"}, {\"id\": 12, \"nam...   \n",
       "\n",
       "                                 homepage     id  \\\n",
       "691  http://video.movies.go.com/wildhogs/  11199   \n",
       "\n",
       "                                              keywords original_language  \\\n",
       "691  [{\"id\": 1599, \"name\": \"midlife crisis\"}, {\"id\"...                en   \n",
       "\n",
       "    original_title                                           overview  \\\n",
       "691      Wild Hogs  Restless and ready for adventure, four suburba...   \n",
       "\n",
       "     popularity                               production_companies  \\\n",
       "691   31.719463  [{\"name\": \"Wild Hogs Productions\", \"id\": 6354}...   \n",
       "\n",
       "                                  production_countries release_date  \\\n",
       "691  [{\"iso_3166_1\": \"US\", \"name\": \"United States o...   2007-03-02   \n",
       "\n",
       "       revenue  runtime                          spoken_languages    status  \\\n",
       "691  253625427    100.0  [{\"iso_639_1\": \"en\", \"name\": \"English\"}]  Released   \n",
       "\n",
       "                                      tagline      title  vote_average  \\\n",
       "691  A lot can happen on the road to nowhere.  Wild Hogs           5.6   \n",
       "\n",
       "     vote_count             extracted_genres  \\\n",
       "691         648  [Action, Adventure, Comedy]   \n",
       "\n",
       "                   extracted_production_companies  \\\n",
       "691  [Wild Hogs Productions, Touchstone Pictures]   \n",
       "\n",
       "    extracted_production_countries  \\\n",
       "691     [United States of America]   \n",
       "\n",
       "                                    extracted_keywords  \\\n",
       "691  [midlife crisis, road trip, politically incorr...   \n",
       "\n",
       "    extracted_spoken_languages  \n",
       "691                  [English]  "
      ]
     },
     "execution_count": 483,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check what movie with 0 budget produced the most revenue\n",
    "zero_budget_movies[zero_budget_movies.revenue == zero_budget_movies.revenue.max()]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68e8fc55-44e5-48e1-be44-cb4d7f89ca41",
   "metadata": {},
   "source": [
    "##### According to the dataset a budgetless movie, which produced the most revenue is 'Wild Hogs'. However, it seems unbelievable and I look for answers in Wikipedia.\n",
    "##### According to Wikipedia, Wild Hogs had a way higher budget than zero, to be precise - 60 millions USD.\n",
    "\n",
    "[Source](https://en.wikipedia.org/wiki/Wild_Hogs)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29df5440-f8a7-4aa8-99c5-b18009f39c28",
   "metadata": {},
   "source": [
    "### Conclusion\n",
    "##### Short answer to the question 'Can budgetless movie sell?': unlikely. \n",
    "##### From the analysis I can speculate that some of the movies did have a higher budget than 0, however, the values were not specified in this datset and simply set to 0. Unfortunately, I cannot determine which movies acctually did not have any budget and which budget's values were missing. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7636d8ce-49d5-4499-950b-6877017ed83b",
   "metadata": {},
   "source": [
    "### Are there any missing values in revenue too?\n",
    "### To investigate this I repeat the process"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 484,
   "id": "97eeb366-79df-49f6-b12d-4c35da543b71",
   "metadata": {},
   "outputs": [],
   "source": [
    "zero_revenue_movies = df[df.revenue == 0] #Create a dataset only with movies, which revenue is set to 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 485,
   "id": "eb55191b-3b59-4258-adbb-9db81c5ef3f6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>budget</th>\n",
       "      <th>genres</th>\n",
       "      <th>homepage</th>\n",
       "      <th>id</th>\n",
       "      <th>keywords</th>\n",
       "      <th>original_language</th>\n",
       "      <th>original_title</th>\n",
       "      <th>overview</th>\n",
       "      <th>popularity</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>production_countries</th>\n",
       "      <th>release_date</th>\n",
       "      <th>revenue</th>\n",
       "      <th>runtime</th>\n",
       "      <th>spoken_languages</th>\n",
       "      <th>status</th>\n",
       "      <th>tagline</th>\n",
       "      <th>title</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>extracted_genres</th>\n",
       "      <th>extracted_production_companies</th>\n",
       "      <th>extracted_production_countries</th>\n",
       "      <th>extracted_keywords</th>\n",
       "      <th>extracted_spoken_languages</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>135</th>\n",
       "      <td>150000000</td>\n",
       "      <td>[{\"id\": 18, \"name\": \"Drama\"}, {\"id\": 27, \"name...</td>\n",
       "      <td>http://www.thewolfmanmovie.com/</td>\n",
       "      <td>7978</td>\n",
       "      <td>[{\"id\": 494, \"name\": \"father son relationship\"...</td>\n",
       "      <td>en</td>\n",
       "      <td>The Wolfman</td>\n",
       "      <td>Lawrence Talbot, an American man on a visit to...</td>\n",
       "      <td>21.214571</td>\n",
       "      <td>[{\"name\": \"Universal Pictures\", \"id\": 33}, {\"n...</td>\n",
       "      <td>[{\"iso_3166_1\": \"US\", \"name\": \"United States o...</td>\n",
       "      <td>2010-02-11</td>\n",
       "      <td>0</td>\n",
       "      <td>102.0</td>\n",
       "      <td>[{\"iso_639_1\": \"en\", \"name\": \"English\"}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>When the moon is full the legend comes to life</td>\n",
       "      <td>The Wolfman</td>\n",
       "      <td>5.5</td>\n",
       "      <td>549</td>\n",
       "      <td>[Drama, Horror, Thriller]</td>\n",
       "      <td>[Universal Pictures, Stuber Productions, Relat...</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[father son relationship, victorian england, r...</td>\n",
       "      <td>[English]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        budget                                             genres  \\\n",
       "135  150000000  [{\"id\": 18, \"name\": \"Drama\"}, {\"id\": 27, \"name...   \n",
       "\n",
       "                            homepage    id  \\\n",
       "135  http://www.thewolfmanmovie.com/  7978   \n",
       "\n",
       "                                              keywords original_language  \\\n",
       "135  [{\"id\": 494, \"name\": \"father son relationship\"...                en   \n",
       "\n",
       "    original_title                                           overview  \\\n",
       "135    The Wolfman  Lawrence Talbot, an American man on a visit to...   \n",
       "\n",
       "     popularity                               production_companies  \\\n",
       "135   21.214571  [{\"name\": \"Universal Pictures\", \"id\": 33}, {\"n...   \n",
       "\n",
       "                                  production_countries release_date  revenue  \\\n",
       "135  [{\"iso_3166_1\": \"US\", \"name\": \"United States o...   2010-02-11        0   \n",
       "\n",
       "     runtime                          spoken_languages    status  \\\n",
       "135    102.0  [{\"iso_639_1\": \"en\", \"name\": \"English\"}]  Released   \n",
       "\n",
       "                                            tagline        title  \\\n",
       "135  When the moon is full the legend comes to life  The Wolfman   \n",
       "\n",
       "     vote_average  vote_count           extracted_genres  \\\n",
       "135           5.5         549  [Drama, Horror, Thriller]   \n",
       "\n",
       "                        extracted_production_companies  \\\n",
       "135  [Universal Pictures, Stuber Productions, Relat...   \n",
       "\n",
       "    extracted_production_countries  \\\n",
       "135     [United States of America]   \n",
       "\n",
       "                                    extracted_keywords  \\\n",
       "135  [father son relationship, victorian england, r...   \n",
       "\n",
       "    extracted_spoken_languages  \n",
       "135                  [English]  "
      ]
     },
     "execution_count": 485,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check what movie with 0 budget produced the most revenue\n",
    "zero_revenue_movies[zero_revenue_movies.budget == zero_revenue_movies.budget.max()] "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3be116eb-fd54-471c-aceb-0428d50d9605",
   "metadata": {},
   "source": [
    "##### According to the data, a movie, which had the largest budget and 0 revenue was 'The Wolfman'. Wikipedia says that, althougth it wasn't profitable, the movie made 142.6 millions USD in revenue. \n",
    "\n",
    "[Source](https://en.wikipedia.org/wiki/The_Wolfman_(film))\n",
    "\n",
    "##### It is evident that in this dataset both budget and revenue have missing values set to 0, which prevents us from analysing it further.  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "768021ed-82c3-4e43-b6ca-acff6b0804b8",
   "metadata": {},
   "source": [
    "## 3. What is the dispersion between women and non-women directed movies?\n",
    "\n",
    "### The question is important to analyse because women involvment in high-qualification positions within the movie industry has always been questionable."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b90e06b1-4bcb-457a-ae60-aaf16f81d40f",
   "metadata": {},
   "source": [
    "### Looking for woman-directed movies"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76e9d460-b942-48af-9460-1672a0e40bf7",
   "metadata": {},
   "source": [
    "##### The women-directed movies' analysis idea came from exploration of keywords in the dataset. It is important to note that one of the most prominent limitations in women-directed movie analysis is that we cannot assume that all movies, which were directed by women, appear in keywords. Therefore, this dataset might have some more women-directed movies.\n",
    "##### At first I want to see how many occurences of women director keyword can be found in the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 486,
   "id": "e931d6f2-8721-463b-aac0-75d2497e1135",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "woman director          324\n",
      "independent film        318\n",
      "duringcreditsstinger    307\n",
      "based on novel          197\n",
      "murder                  189\n",
      "aftercreditsstinger     170\n",
      "violence                150\n",
      "dystopia                139\n",
      "sport                   126\n",
      "revenge                 118\n",
      "Name: extracted_keywords, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# The 'explode' function is used to expand a column that contains lists or arrays into multiple rows,\n",
    "# where each element in the list or array gets its own row. In this case, the 'extracted_keywords' column is exploded.\n",
    "exploded = df.explode('extracted_keywords')\n",
    "\n",
    "# The 'value_counts' function is used to count the occurrences of each unique element in the 'extracted_keywords' column.\n",
    "# It returns a Series object with the elements as the index and their respective counts as the values.\n",
    "\n",
    "counts = exploded['extracted_keywords'].value_counts()\n",
    "\n",
    "# The 'head' function is used to display the first few rows of the 'counts' Series, with a default of 5 rows.\n",
    "# By passing the argument '10' to 'head', it displays the first 10 rows of the Series.\n",
    "# This line prints the top 10 most frequent keywords and their corresponding counts.\n",
    "print(counts.head(10))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55b90191-2ea3-4342-bd5c-74673ab6b8d9",
   "metadata": {},
   "source": [
    "##### The dataset has 324 movies with keyword - 'woman_directed', it is the most frequent keyword."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9bf39ee1-4573-4aad-b6aa-02c2493bf187",
   "metadata": {},
   "source": [
    "##### Since there is no column, which directly measures, whether a movie was directed by a woman, I attempt to retrieve it from movies' keywords."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "af232bee-356f-4144-8f1d-2bda6b372fd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# The line adds a new column 'is_woman_director' to the DataFrame 'df' based on the values in the 'extracted_keywords' column.\n",
    "# The 'apply' function is used to apply a lambda function to each value in the 'extracted_keywords' column.\n",
    "\n",
    "# The lambda function checks if the string 'woman director' is present in each value of 'extracted_keywords'.\n",
    "# If 'woman director' is found, it returns 1; otherwise, it returns 0.\n",
    "\n",
    "# The result of the lambda function is assigned to the 'is_woman_director' column, indicating whether a movie has a woman director.\n",
    "df['is_woman_director'] = df['extracted_keywords'].apply(lambda x: 1 if 'woman director' in x else 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fde64310-9ee9-40e7-a8d8-427d422ba323",
   "metadata": {},
   "source": [
    "### How does women-director distribution look like over time? "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3d049d32-d884-4def-9a2d-c0dd7156f481",
   "metadata": {},
   "source": [
    "##### At first I extract the year from movies' release date to make the following plots more readable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "721b243d-7291-4b12-aa81-6989aa99ce56",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Change the release_date data type to date (it was object).\n",
    "df.release_date = df.release_date.astype('datetime64')\n",
    "# Extract year from release_date and adding it to the dataset\n",
    "df['release_year'] = pd.to_datetime(df['release_date']).dt.year"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "109bae15-d9b9-4203-a0a5-a77c39c0cced",
   "metadata": {},
   "source": [
    "##### Plotting the ocurrences, when movie director was a woman over the years."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "b2adc71f-a669-486d-83dd-d6b9f01d387f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x='release_year', # The 'release_year' column is assigned to the x-axis \n",
    "                y='is_woman_director', # The 'is_woman_director' column is assigned to the y-axis\n",
    "                data=df, # The data is provided from the DataFrame 'df'\n",
    "                color='purple') # The scatterplot is set to be purple in color.\n",
    "\n",
    "# Set the x-axis label to 'Release Year'.\n",
    "plt.xlabel('Year')\n",
    "\n",
    "# Set the y-axis label to 'Is Woman Director'.\n",
    "plt.ylabel('Is Woman Director: 0-No, 1-Yes')\n",
    "\n",
    "# Set the title of the plot to 'Distribution of Woman Directors by Release Year'.\n",
    "plt.title('Distribution of Woman Directors by Release Year')\n",
    "\n",
    "# Display the plot.\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72769e76-444f-4ae0-a971-f179f2cb30c5",
   "metadata": {},
   "source": [
    "##### From the plot it is possible to notice that this dataset has no woman-director keywords for movies before 1980. It is likely that there aren't any women directed movies before that date (in this dataset)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c571dba3-906f-4d86-9b70-c22d7c70ec25",
   "metadata": {},
   "source": [
    "### Has the situation changed in this century?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1490dd43-09d3-4626-8013-4b690133ec57",
   "metadata": {},
   "source": [
    "##### Since there are no movies in this dataset, which had a keyword 'woman-director' before 1980, I define a dataset between year 1980 and 1999. It will represent women involvment in the previous century. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "529762bf-c72d-490f-851e-768328095848",
   "metadata": {},
   "outputs": [],
   "source": [
    "# The line creates a new DataFrame 'df1' by filtering the original DataFrame 'df' based on certain conditions.\n",
    "\n",
    "# The conditions are specified within the square brackets [].\n",
    "# It checks if the values in the 'release_year' column are greater than 1980 and less than 2000.\n",
    "\n",
    "# The '&' operator is used to combine the two conditions, indicating that both conditions must be true for a row to be included in 'df1'.\n",
    "\n",
    "# The resulting DataFrame 'df1' contains rows where the 'release_year' is between 1980 and 2000 (exclusive).\n",
    "df1 = df[(df.release_year > 1980) & (df.release_year < 2000)] "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "27cf2b0c-8f7d-4446-9ebe-90aaac3f65eb",
   "metadata": {},
   "source": [
    "##### Visualizing the percentage of woman-director keyword in the dataset between year 1980-1999"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "dd82a6a7-58ad-492d-88eb-60f541e81be1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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30/BggXp2bMna9asYebMmRw5cgQAf39/3nzzTQIDA3FycuL69eukpKTQqVMnduzYQXh4OJ9//jnlypV75OvxsHVN7+LFi4wbNw43Nzfc3NwoUqQIAH5+fuzbt4+AgACKFSvG3bt3+eqrr5g8eTJXr15F0zSGDRtG48aN2bFjhzXwn332WXr27MmePXs4efIkVapU4fDhw3z//fe4uLhQoUIFpkyZwoYNG1i1ahWapjF06FAWLFjw0O0nMTGRwYMH07VrV7p06cLcuXM5dOgQSin69etH06ZN6datG1u2bMFgMPDvf/+bYsWKcejQIb788ks2btzIV199xfr16zl8+DDr1q1j1KhRjBo1iri4OMxmMx988AG+vr74+/tToUIFXFxcCAwMZOTIkSilKFGiRM59IIRITwlhB06dOvXQxz/88EO1Y8cOpZRSvXr1UseOHVN9+/ZVa9asUUoptXTpUjV16lR14MAB1bFjR5WcnKwSExNVmzZt1K1bt1SPHj3U+fPnlVJKrVixQn3yyScqNDRUrVixQimlVFJSkmrUqJFSSqkxY8ao//znP0oplelzFi9erJRS6scff1QTJkxQt27dUm3btlWJiYnKbDar6dOnq2vXrqmOHTuqhIQEpZRSI0eOVHv37s2wbt26dbOu99atW9WQIUOUUkq1bNlSJSUlPfT1CAwMVIcOHVIrV65UnTt3Vps2bVI7d+5U//73v9X27dvVoEGDlKZpKiUlRb366qvqzJkzasyYMerzzz9XSik1YcIENWvWLKWUUvPnz1f/+c9/Hmtd7zdkyBDren355ZdqzJgxSimlmjRpopRSqm/fvurXX39VSin1ww8/qNmzZyullIqOjladOnVSqampqmXLlurWrVtKKaUWLlyorl27psaMGaN27dqloqOjVZs2bVRsbKxSSqnp06eroKAgtWrVKvXee+899DVSSqkDBw6ogQMHqjfeeEP99ttvSimldu7cqYYNG2Zdzy5duqg7d+6o0aNHq507dyqTyaT8/f1VcnKy8vf3V0lJSWr06NGqS5cuKjIyUs2aNUvt2rVLzZw5U3333XdKKaVu3LihWrZsqcxms2rZsqU6efKkUkqpmTNnqh9//FEppdSmTZtU3759H1rno7Z7IbJDWuDCbt25c4fdu3cTHR1NUFAQcXFxLF26FIAXXngBgHr16rF9+3YA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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Count the occurrences of each category in the 'is_woman_director' column in the filtered DataFrame 'df1'.\n",
    "\n",
    "counts1 = df1['is_woman_director'].value_counts()\n",
    "\n",
    "# Create a pie chart \n",
    "\n",
    "plt.pie(counts1, \n",
    "        labels=counts1.index, # The 'counts' series is used as the data, labels are set to 'counts.index' (the categories)\n",
    "        autopct='%1.1f%%', # Autopct specifies the percentage format.\n",
    "        colors=['pink', 'brown']) # Mmake the slices pink and purple.\n",
    "\n",
    "\n",
    "plt.legend(counts1.index, # Add a legend using the categories from 'counts.index'.\n",
    "           title=\"Appearance of 'woman-director' keyword\")# The title of the legend is set to \"Appearance of 'woman director' keyword\".\n",
    "\n",
    "# Set the title of the pie chart.\n",
    "\n",
    "plt.title(\"Distribution of movies with and without 'woman director' keyword over the years 1980-1999\")\n",
    "\n",
    "\n",
    "# Display the chart.\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "789810d7-4f5a-4d4c-b00a-674f6705ea92",
   "metadata": {},
   "source": [
    "##### From the graph we can see that the proportion of movies, having woman-directed keyword, between years 1980 and 1999 is 5 %. \n",
    "##### Now i want to compare the percantage with one between years 2000 and 2017 (2017 is the last release year in the dataset). It will represent woman-director keyword appearance in this century."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "2cf4c4b3-4300-4fe0-b96a-d1b98e024cc7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# The line creates a new DataFrame 'df2' by filtering the original DataFrame 'df' based on a condition.\n",
    "\n",
    "# The condition is specified within the square brackets [].\n",
    "# It checks if the values in the 'release_year' column are greater than or equal to 2000.\n",
    "\n",
    "# The resulting DataFrame 'df2' contains rows where the 'release_year' is greater than or equal to 2000.\n",
    "\n",
    "df2 = df[df.release_year >= 2000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "25da5320-bc6e-44be-9cae-b4aef970cff9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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swCefPLJ9KysrJzMzMzcK664olv5NFu1ajXsuuuu6zp48OABH374YZv27dsHExMTqz3WZffu3WbMmDGZU6dO7VRdpuHDh/f/9NNPW5Y/5/DDDx8wd+7clgfzvkTOi927d5tTTz21T1ZWVs4pp5zSp7i42FSXV6RBWGv1p7+Y/3399dcrrbWByL9HH330h4kTJ2621gaGDh26+5NPPllsrQ289dZbS1NSUsIFBQULysrKAqNHjy585plnvrfWBgD72GOP/WCtDUyePHntpEmTNllrA6NGjdq5YMGChdbawIcffvjtyJEjd1prA5s2bZofCoUC1trA1KlTV1588cUbrLWB66+/fl1OTs6eXbt2zbPWBnbu3PnVnj175llrAwsWLFiYm5u7pzxLmzZtgsuXL/8mGAwGhgwZsvvdd99dsnfv3nndunUr+fjjjxdbawNbt279qrS0NPDggw+uvOmmm9ZaawNFRUXzcnNz93z77bcLIl/37NmzF2dmZhYVFhZ+tWPHjq/69u27d86cOQXW2kDXrl1L1q1b93XleRX5d+65526aOXPmd1988cWi3NzcPWefffZma22gR48exTt27Pjq9ttvXz1hwoQt1trAV199tejQQw8t2bNnz7yHHnpoRffu3Yu3bdv21dq1a79u06ZN8IEHHvjRWhu46KKLNt51112rrLWB9evXzy+f1jXXXLNuypQpq6y1gTPOOGPLuHHjtgWDwUAgEFjUvXv34qryHXvssTsefvjhFdbawL333vtjy5YtQ9bawJIlSxb069dvr7U28NBDD63o0qVL6YYNG+ZbawNXXXXV+kcfffQHa21g8+bN83v27FlcWFj41f333//jz3/+8+2lpaUBa22gfPzI+bRixYpvDj300JK1a9d+XVpaGhg5cuTO559/fnn58jJ9+vTva5qf5e0tWbJkwejRo3eWZ68u07Rp01ZceOGFG621gW+++aZiWTmY9yVyXtxxxx2rzzzzzC3W2sDnn39ekJiYaP1+/+LKmd3PlOefbf01/T/1GEiDeeWVV9LPPffc7QATJkzYFrk7YdCgQXtycnJKk5KSOOuss7Z98sknbQASEhK4+OKLtwFcdNFFW7/44os2hYWFCfPnz28zceLEvgMGDMi58sore27atCkZYMWKFSljxozJzMrKypk2bdqhS5YsqdjSGzdu3I42bdpYgNLSUpOfn98rKysrZ+LEiX2///77FpFZ+vbtW5aYmEhubm7R999/n7JgwYIWXbp0KfP5fEUA6enp4eTkZD744IO2r7zySscBAwbkDBs2LHv79u1Jixcv3u/4iY8//rjNySefvKNt27bhdu3ahU855ZTtH330UVq0823MmDG7/X5/mw8//DDt4osv3vTtt9+2XLFiRXK7du2C7dq1C3/66adtfv3rX28FGDZsWHHXrl1LFy5c2ALgyCOP3NWhQ4dw165dg23atAlNnDhxh/sai1auXJkKMG/evJbDhw/vn5WVlfP66693LCgoqMg/fvz4HYmJiQwfPrx469atyVXl++qrr9pccskl2wAuu+yyrTW8jp2HHHJIyJ0nbf/yl78cNmDAgJyjjz66f0lJiVm+fHnK//3f/7W9/PLLNycnO5MqHz/SnDlzWo8aNWpX165dg8nJyZx99tnb/H5/G4DExEQuuOCC7dHM1/Hjx/ebNGnSlquvvnprTZkuuOCC7R988EG7kpIS88QTT3TKz8/fcrDvS+S8mDNnTptJkyZtBRg5cuTerKysomjyi9SXRnnwjTQ9GzZsSPz888/bLlu2rOXVV19NKBQyxhj7+OOPrwEwZv/e08r3Ix8PhUKkpaUFlyxZsrjy8KuvvrrHtddeu+G8884rnDVrVtrdd99d0bXdunXrcPnte+6555AuXbqUvf766yvC4TAtW7YcXj4sNTW1ous5MTGRYDBorLUYY37SJW2tNVOnTl01YcKEndW9dmsP7qzNE044YddTTz3VZc2aNSUPPPDA2jfffLPDCy+80GHUqFG7a2s/JSWlYmBCQgItWrSw5beDwaABuPTSS3u/9tpry0ePHr132rRpHf1+f0XRUj5+5HSuueaabu+//347gPL3IJpTU1u1alUx/621vPbaa8uHDBlSEjlOdfO58jg1vN5wUlJ0q7Ujjjhi97vvvtvusssu25aQkFBtJnC+yF966aX2b775Zvq8efMWw8G9L5HzAqpf3kW8oB4DaRAzZszocMYZZ2xdt27dwrVr1y7csGHDgoyMjNL33nuvDcDChQtbL1myJCUUCvHaa6+ljxkzZhdAOBymfN/2c88913HEiBG70tPTwxkZGaXPPPNMh/JxPvvss5YAu3btSuzRo0dZ+fjV5SksLEw87LDDyhITE3nsscc6lh/TUJ0hQ4YUb9y4McXv97cC2L59e0JZWRknnHBC4eOPP965pKTEACxYsCB1586d+32ujjvuuN3vvPNO+127diXs3Lkz4Z133ulw7LHH7op23vXr169s+/btSStWrGiRk5NTOnr06N2PPvroocccc8xugKOPPnr3Cy+8kF4+/fXr16cMHjy4ONr2i4qKEnr06FFWUlJiXn755VoPCn344YfXLlmyZHF5UXD44Yfvnj59ejrA9OnTq53nkY499tidU6dOPaT8+I3//ve/LQGOP/74nU888UTn8oPyyo+DaN26daiwsDAB4Jhjjtkzd+7ctPXr1ycFg0FeffXV9LFjx+6O9vWWe/DBB9elp6cHJ02a1KOmTACXX375lptvvrn7kCFD9pRv6cfqfYkc78svv2yxbNkynZEgnlJhIA3i1Vdf7XjGGWfs18V72mmnbS/fnTB06NDdkydPzsjKysrt0aNHyaRJk3YAtGzZMlxQUNAyNzc3e/bs2Wn33XffeoCZM2f+8Oyzz3bq379/TmZmZu7rr7/eHuDWW29dd+655/YdPnx4/44dO1Z7WPp11123aebMmR2HDBkyYNmyZS1atmwZrm5ccLacX3zxxe9/+9vf9ujfv3/O2LFjs4qKihKuv/76LQMGDCgeNGhQdmZmZu4ll1zSs6ysbL/Nv6OPProoPz9/6+GHH549fPjw7EmTJm0+6qij9tZl/g0dOnRP7969iwHGjh27a9OmTcnHH3/8LoDf/e53m0KhkMnKyso5++yz+z755JMrW7ZsGXU3xS233LJuxIgR2WPGjMnKzMyMuqAo99hjj6166qmnugwcODC7sLAwMZrn3H///euCwaAZMGBATmZmZu5tt93WDeD666/fnJGRUTpgwIDc/v375zz99NPpAOeff/6Wk046KXPkyJFZPXv2LLv99tvX+ny+rOzs7NzBgwcX/epXv9pR19wATz/99OqSkpKEyy+/PKO6TABjxowpat26dejCCy/cEvn8WLwvN95446Y9e/YkZmVl5dx7772HDho0aM+BvBaRWDEH280pUpVvvvlm5ZAhQ7bUPqZzJsDUqVMP+eijj5ZXHtaqVathRUVF82OfUCR6K1euTB47dmz/77//flFiYlS1T8x98803nYYMGdLLk4lLs6IeAxGRGjzyyCMdR40alX377bev9aooEGlI6jGQelGXHgMRqZ16DKSh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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Count the occurrences of each category in the 'is_woman_director' column in the filtered DataFrame 'df1'.\n",
    "\n",
    "counts2 = df2['is_woman_director'].value_counts()\n",
    "\n",
    "# Create a pie chart \n",
    "\n",
    "plt.pie(counts2, \n",
    "        labels=counts2.index, # The 'counts' series is used as the data, labels are set to 'counts.index' (the categories)\n",
    "        autopct='%1.1f%%', # Autopct specifies the percentage format.\n",
    "        colors=['pink', 'brown']) # Mmake the slices pink and purple.\n",
    "\n",
    "\n",
    "plt.legend(counts2.index, # Add a legend using the categories from 'counts.index'.\n",
    "           title=\"Appearance of 'woman-director' keyword\")# The title of the legend is set to \"Appearance of 'woman director' keyword\".\n",
    "\n",
    "# Set the title of the pie chart.\n",
    "\n",
    "plt.title(\"Distribution of movies with and without 'woman director' over the years 2000-2017\")\n",
    "\n",
    "\n",
    "# Display the chart.\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7b350f01-b479-4fe0-9e46-5e0026b7ad84",
   "metadata": {},
   "source": [
    "##### In the years between 2000 and 2017 the proportion of movies with 'woman-directed' keyword has grown slightly compared with the years 1980-1999. It now stands at 7.8%."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9dd4926-afa7-4baf-b53d-733baed778e1",
   "metadata": {},
   "source": [
    "### Concusion\n",
    "##### To answer the question 'What is the dispersion between woman and non-woman directed movies?' I cannot answer with certainty, having results from this dataset, because it is likely that some woman-directed movies simply did not have a keyword 'woman director'. Moreover, I cannot generalize the answer, since the dataset does not include all or at least a major proportion of total movies.\n",
    "\n",
    "##### Looking at the percentage of woman-directed keyword in movies from this century it is somewhat consistent with the past studies. A research on the dispersion of woman and man directors between 2007 and 2023 reveal that the proportion of woman directors is 6 %.\n",
    "\n",
    "[Source](https://annenberg.usc.edu/news/research-and-impact/was-2023-year-woman-director-survey-says…-no)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "602d5c09-463b-4ba9-90bd-e1933fe239e7",
   "metadata": {},
   "source": [
    "## 4. Are woman-directed movies less popular?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "03b8ed24-15d0-4345-b93f-f41ac8e49c12",
   "metadata": {},
   "source": [
    "##### The recognition of a movie can be partially determined by its popularity among the population. Analyzing the popularity of woman-directed movies we can check if movies directed by women are recognized by the public similarly to non-woman. \n",
    "##### To investigate the last 2 questions I will use the first movie in this dataset, which has woman-director keyword and the latest one. \n",
    "### Looking for the first movie, which has a 'woman-directed' keyword"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "5ee640f3-c56c-4140-b14a-bdba8f03d217",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>budget</th>\n",
       "      <th>genres</th>\n",
       "      <th>homepage</th>\n",
       "      <th>id</th>\n",
       "      <th>keywords</th>\n",
       "      <th>original_language</th>\n",
       "      <th>original_title</th>\n",
       "      <th>overview</th>\n",
       "      <th>popularity</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>...</th>\n",
       "      <th>title</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>extracted_genres</th>\n",
       "      <th>extracted_production_companies</th>\n",
       "      <th>extracted_production_countries</th>\n",
       "      <th>extracted_keywords</th>\n",
       "      <th>extracted_spoken_languages</th>\n",
       "      <th>is_woman_director</th>\n",
       "      <th>release_year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2249</th>\n",
       "      <td>0</td>\n",
       "      <td>[{\"id\": 35, \"name\": \"Comedy\"}, {\"id\": 10402, \"...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>40932</td>\n",
       "      <td>[{\"id\": 11663, \"name\": \"camp\"}, {\"id\": 155490,...</td>\n",
       "      <td>en</td>\n",
       "      <td>Can't Stop the Music</td>\n",
       "      <td>Movie about the Village People filmed in a doc...</td>\n",
       "      <td>6.574285</td>\n",
       "      <td>[{\"name\": \"EMI Films\", \"id\": 8263}]</td>\n",
       "      <td>...</td>\n",
       "      <td>Can't Stop the Music</td>\n",
       "      <td>4.9</td>\n",
       "      <td>9</td>\n",
       "      <td>[Comedy, Music]</td>\n",
       "      <td>[EMI Films]</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[camp, disco, woman director, village people]</td>\n",
       "      <td>[English]</td>\n",
       "      <td>1</td>\n",
       "      <td>1980</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 27 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      budget                                             genres homepage  \\\n",
       "2249       0  [{\"id\": 35, \"name\": \"Comedy\"}, {\"id\": 10402, \"...      NaN   \n",
       "\n",
       "         id                                           keywords  \\\n",
       "2249  40932  [{\"id\": 11663, \"name\": \"camp\"}, {\"id\": 155490,...   \n",
       "\n",
       "     original_language        original_title  \\\n",
       "2249                en  Can't Stop the Music   \n",
       "\n",
       "                                               overview  popularity  \\\n",
       "2249  Movie about the Village People filmed in a doc...    6.574285   \n",
       "\n",
       "                     production_companies  ...                 title  \\\n",
       "2249  [{\"name\": \"EMI Films\", \"id\": 8263}]  ...  Can't Stop the Music   \n",
       "\n",
       "     vote_average  vote_count  extracted_genres  \\\n",
       "2249          4.9           9   [Comedy, Music]   \n",
       "\n",
       "     extracted_production_companies extracted_production_countries  \\\n",
       "2249                    [EMI Films]     [United States of America]   \n",
       "\n",
       "                                 extracted_keywords  \\\n",
       "2249  [camp, disco, woman director, village people]   \n",
       "\n",
       "     extracted_spoken_languages  is_woman_director  release_year  \n",
       "2249                  [English]                  1          1980  \n",
       "\n",
       "[1 rows x 27 columns]"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# The first condition checks if the values in the 'is_woman_director' column are equal to 1, indicating a woman director.\n",
    "# The second condition checks if the values in the 'release_year' column are equal to 1980.\n",
    "# Both conditions must be met.\n",
    "df[(df.is_woman_director == 1) & (df.release_year == 1980)]\n",
    "\n",
    "#In this dataset, according to release date the first movie, directed by a woman was in 1980"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "83808cc4-8040-4238-b72b-37653e52054a",
   "metadata": {},
   "source": [
    "##### The first movie is 'Can't Stop the feeling' 1980, genres - Comedy and Music, popularity ~ 6.57. It would be interesting to compare this movie with others from the same year and similar genre to see the popularity of woman directed movie in the bigger picture."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "1f94eb78-da82-49d5-b596-7e9dff55ba0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filter the DataFrame 'df' for rows where 'id' is equal to 40932.\n",
    "woman_dir_movie = df[df.id == 40932]\n",
    "\n",
    "# Filter the DataFrame 'df' for rows where 'release_year' is 1980, 'is_woman_director' is 0 (no woman director), and 'extracted_genres' contains 'Comedy'.\n",
    "comedy = df[(df.release_year == 1980) & (df.is_woman_director == 0) & (df['extracted_genres'].apply(lambda x: any('Comedy' in genre for genre in x)))]\n",
    "\n",
    "# Filter the DataFrame 'df' for rows where 'release_year' is 1980, 'is_woman_director' is 0 (no woman director), and 'extracted_genres' contains 'Music'.\n",
    "\n",
    "music = df[(df.release_year == 1980) & (df.is_woman_director == 0) & (df['extracted_genres'].apply(lambda x: any('Music' in genre for genre in x)))]\n",
    "\n",
    "# Concatenate the DataFrames 'music', 'comedy', and 'woman_dir_movie' vertically.\n",
    "# Sort the resulting DataFrame 'joined_df' by the 'popularity' column in ascending order.\n",
    "joined_df = pd.concat([music, comedy, woman_dir_movie], axis=0).sort_values(by='popularity', ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d220a874-c580-4f9e-b0fd-c737f06451c1",
   "metadata": {},
   "source": [
    "##### Since now we have a dataframe with all movies in Comedy and/or Music genres from 1980s, I plot their popularity\n",
    "### How a woman-directed movie compares with similar non-woman directed movies in terms of popularity?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "174530ae-e46b-443f-8a5d-6b80f68e7c46",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Set the plot style\n",
    "sns.set_style('whitegrid')\n",
    "\n",
    "fig, ax = plt.subplots()\n",
    "\n",
    "# Plot the budget bars\n",
    "sns.barplot(x= 'title', y='popularity', data=joined_df, color= 'pink', ax=ax)\n",
    "\n",
    "# Calculate the mean popularity\n",
    "mean_popularity = df[df.release_year == 1980].popularity.mean()\n",
    "\n",
    "# Add the mean popularity line\n",
    "ax.axhline(mean_popularity, color='purple', linestyle='--', label='Mean Popularity')\n",
    "\n",
    "\n",
    "# Add labels and title\n",
    "ax.set_xlabel('Movie Title')\n",
    "ax.set_ylabel('Popularity')\n",
    "ax.set_title('Popularity of Comedy and/or Music genre movies in 1980')\n",
    "\n",
    "#Set axis labels rotation\n",
    "plt.xticks(rotation=60) \n",
    "\n",
    "#Add a legend\n",
    "plt.legend()\n",
    "\n",
    "# Display the chart\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2af139f7-b4a0-42f0-b89d-88267cb64fda",
   "metadata": {},
   "source": [
    "##### From the graph, movie 'Can't stop the music' stands in the 5th place out of 7 by popularity. It is also way behind the mean (~21.5) movie popularity for the year 1980."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8dc8af72-f57b-416a-8fee-db92f3f91df6",
   "metadata": {},
   "source": [
    "### Conclusion: \n",
    "##### To answer the question 'Are woman-dirceted movies less popular?' this dataset in insufficient to determine the effects of being a woman director.\n",
    "##### However, having the historical context and the graph above, I can conclude that in this dataset woman-directed Comedy and/or Music genre movie from the 1980 was less popular and bellow the mean popularity compared to the rest."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99e3ed94-5603-4dad-a503-b5f5ce2dcfc0",
   "metadata": {},
   "source": [
    "## 5. Do women directed movies receive less budget? What about revenue?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89df51ce-baf5-4ea1-b41d-f300845064c3",
   "metadata": {},
   "source": [
    "##### Now I investigate the latest woman-directed movie to see how it compares with non-woman directed movies in terms of budget and revenue."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eac4da1a-218e-492a-aee0-6ff44ce4af56",
   "metadata": {},
   "source": [
    "### Looking of the last woman-directed movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "055efd2e-f367-4cfd-a5a6-3ff4ce54b230",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>budget</th>\n",
       "      <th>genres</th>\n",
       "      <th>homepage</th>\n",
       "      <th>id</th>\n",
       "      <th>keywords</th>\n",
       "      <th>original_language</th>\n",
       "      <th>original_title</th>\n",
       "      <th>overview</th>\n",
       "      <th>popularity</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>...</th>\n",
       "      <th>title</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>extracted_genres</th>\n",
       "      <th>extracted_production_companies</th>\n",
       "      <th>extracted_production_countries</th>\n",
       "      <th>extracted_keywords</th>\n",
       "      <th>extracted_spoken_languages</th>\n",
       "      <th>is_woman_director</th>\n",
       "      <th>release_year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>152</th>\n",
       "      <td>145000000</td>\n",
       "      <td>[{\"id\": 28, \"name\": \"Action\"}, {\"id\": 12, \"nam...</td>\n",
       "      <td>http://www.kungfupanda.com/</td>\n",
       "      <td>140300</td>\n",
       "      <td>[{\"id\": 478, \"name\": \"china\"}, {\"id\": 779, \"na...</td>\n",
       "      <td>en</td>\n",
       "      <td>Kung Fu Panda 3</td>\n",
       "      <td>Continuing his \"legendary adventures of awesom...</td>\n",
       "      <td>56.747978</td>\n",
       "      <td>[{\"name\": \"Twentieth Century Fox Film Corporat...</td>\n",
       "      <td>...</td>\n",
       "      <td>Kung Fu Panda 3</td>\n",
       "      <td>6.7</td>\n",
       "      <td>1603</td>\n",
       "      <td>[Action, Adventure, Animation, Comedy, Family]</td>\n",
       "      <td>[Twentieth Century Fox Film Corporation, Dream...</td>\n",
       "      <td>[China, United States of America]</td>\n",
       "      <td>[china, martial arts, kung fu, village, panda,...</td>\n",
       "      <td>[English]</td>\n",
       "      <td>1</td>\n",
       "      <td>2016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1770</th>\n",
       "      <td>27000000</td>\n",
       "      <td>[{\"id\": 53, \"name\": \"Thriller\"}]</td>\n",
       "      <td>NaN</td>\n",
       "      <td>303858</td>\n",
       "      <td>[{\"id\": 258, \"name\": \"bomb\"}, {\"id\": 1589, \"na...</td>\n",
       "      <td>en</td>\n",
       "      <td>Money Monster</td>\n",
       "      <td>Financial TV host Lee Gates and his producer P...</td>\n",
       "      <td>38.279458</td>\n",
       "      <td>[{\"name\": \"TriStar Pictures\", \"id\": 559}, {\"na...</td>\n",
       "      <td>...</td>\n",
       "      <td>Money Monster</td>\n",
       "      <td>6.5</td>\n",
       "      <td>1068</td>\n",
       "      <td>[Thriller]</td>\n",
       "      <td>[TriStar Pictures, Sony Pictures Releasing, Sm...</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[bomb, sniper, tv show, hostage drama, police,...</td>\n",
       "      <td>[English]</td>\n",
       "      <td>1</td>\n",
       "      <td>2016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2035</th>\n",
       "      <td>0</td>\n",
       "      <td>[{\"id\": 53, \"name\": \"Thriller\"}]</td>\n",
       "      <td>NaN</td>\n",
       "      <td>205588</td>\n",
       "      <td>[{\"id\": 818, \"name\": \"based on novel\"}, {\"id\":...</td>\n",
       "      <td>en</td>\n",
       "      <td>Our Kind of Traitor</td>\n",
       "      <td>A young Oxford academic and his attorney girlf...</td>\n",
       "      <td>10.547959</td>\n",
       "      <td>[{\"name\": \"StudioCanal\", \"id\": 694}, {\"name\": ...</td>\n",
       "      <td>...</td>\n",
       "      <td>Our Kind of Traitor</td>\n",
       "      <td>6.0</td>\n",
       "      <td>160</td>\n",
       "      <td>[Thriller]</td>\n",
       "      <td>[StudioCanal, Film4, Anton Capital Entertainme...</td>\n",
       "      <td>[France, United Kingdom]</td>\n",
       "      <td>[based on novel, woman director]</td>\n",
       "      <td>[Pусский, English, Français]</td>\n",
       "      <td>1</td>\n",
       "      <td>2016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2109</th>\n",
       "      <td>20000000</td>\n",
       "      <td>[{\"id\": 18, \"name\": \"Drama\"}, {\"id\": 10749, \"n...</td>\n",
       "      <td>http://www.mebeforeyoumovie.net</td>\n",
       "      <td>296096</td>\n",
       "      <td>[{\"id\": 392, \"name\": \"england\"}, {\"id\": 818, \"...</td>\n",
       "      <td>en</td>\n",
       "      <td>Me Before You</td>\n",
       "      <td>A small town girl is caught between dead-end j...</td>\n",
       "      <td>53.161905</td>\n",
       "      <td>[{\"name\": \"New Line Cinema\", \"id\": 12}, {\"name...</td>\n",
       "      <td>...</td>\n",
       "      <td>Me Before You</td>\n",
       "      <td>7.6</td>\n",
       "      <td>2562</td>\n",
       "      <td>[Drama, Romance]</td>\n",
       "      <td>[New Line Cinema, Sunswept Entertainment, Metr...</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[england, based on novel, depression, small to...</td>\n",
       "      <td>[English]</td>\n",
       "      <td>1</td>\n",
       "      <td>2016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2734</th>\n",
       "      <td>16000000</td>\n",
       "      <td>[{\"id\": 18, \"name\": \"Drama\"}]</td>\n",
       "      <td>http://www.miraclesfromheaven-movie.com/</td>\n",
       "      <td>339984</td>\n",
       "      <td>[{\"id\": 2618, \"name\": \"miracle\"}, {\"id\": 5950,...</td>\n",
       "      <td>en</td>\n",
       "      <td>Miracles from Heaven</td>\n",
       "      <td>A faith based movie. A young girl suffering fr...</td>\n",
       "      <td>10.248439</td>\n",
       "      <td>[{\"name\": \"TriStar Pictures\", \"id\": 559}, {\"na...</td>\n",
       "      <td>...</td>\n",
       "      <td>Miracles from Heaven</td>\n",
       "      <td>6.7</td>\n",
       "      <td>186</td>\n",
       "      <td>[Drama]</td>\n",
       "      <td>[TriStar Pictures, Sony Pictures Entertainment]</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[miracle, christian, cure, woman director, acc...</td>\n",
       "      <td>[English]</td>\n",
       "      <td>1</td>\n",
       "      <td>2016</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 27 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         budget                                             genres  \\\n",
       "152   145000000  [{\"id\": 28, \"name\": \"Action\"}, {\"id\": 12, \"nam...   \n",
       "1770   27000000                   [{\"id\": 53, \"name\": \"Thriller\"}]   \n",
       "2035          0                   [{\"id\": 53, \"name\": \"Thriller\"}]   \n",
       "2109   20000000  [{\"id\": 18, \"name\": \"Drama\"}, {\"id\": 10749, \"n...   \n",
       "2734   16000000                      [{\"id\": 18, \"name\": \"Drama\"}]   \n",
       "\n",
       "                                      homepage      id  \\\n",
       "152                http://www.kungfupanda.com/  140300   \n",
       "1770                                       NaN  303858   \n",
       "2035                                       NaN  205588   \n",
       "2109           http://www.mebeforeyoumovie.net  296096   \n",
       "2734  http://www.miraclesfromheaven-movie.com/  339984   \n",
       "\n",
       "                                               keywords original_language  \\\n",
       "152   [{\"id\": 478, \"name\": \"china\"}, {\"id\": 779, \"na...                en   \n",
       "1770  [{\"id\": 258, \"name\": \"bomb\"}, {\"id\": 1589, \"na...                en   \n",
       "2035  [{\"id\": 818, \"name\": \"based on novel\"}, {\"id\":...                en   \n",
       "2109  [{\"id\": 392, \"name\": \"england\"}, {\"id\": 818, \"...                en   \n",
       "2734  [{\"id\": 2618, \"name\": \"miracle\"}, {\"id\": 5950,...                en   \n",
       "\n",
       "            original_title                                           overview  \\\n",
       "152        Kung Fu Panda 3  Continuing his \"legendary adventures of awesom...   \n",
       "1770         Money Monster  Financial TV host Lee Gates and his producer P...   \n",
       "2035   Our Kind of Traitor  A young Oxford academic and his attorney girlf...   \n",
       "2109         Me Before You  A small town girl is caught between dead-end j...   \n",
       "2734  Miracles from Heaven  A faith based movie. A young girl suffering fr...   \n",
       "\n",
       "      popularity                               production_companies  ...  \\\n",
       "152    56.747978  [{\"name\": \"Twentieth Century Fox Film Corporat...  ...   \n",
       "1770   38.279458  [{\"name\": \"TriStar Pictures\", \"id\": 559}, {\"na...  ...   \n",
       "2035   10.547959  [{\"name\": \"StudioCanal\", \"id\": 694}, {\"name\": ...  ...   \n",
       "2109   53.161905  [{\"name\": \"New Line Cinema\", \"id\": 12}, {\"name...  ...   \n",
       "2734   10.248439  [{\"name\": \"TriStar Pictures\", \"id\": 559}, {\"na...  ...   \n",
       "\n",
       "                     title vote_average  vote_count  \\\n",
       "152        Kung Fu Panda 3          6.7        1603   \n",
       "1770         Money Monster          6.5        1068   \n",
       "2035   Our Kind of Traitor          6.0         160   \n",
       "2109         Me Before You          7.6        2562   \n",
       "2734  Miracles from Heaven          6.7         186   \n",
       "\n",
       "                                    extracted_genres  \\\n",
       "152   [Action, Adventure, Animation, Comedy, Family]   \n",
       "1770                                      [Thriller]   \n",
       "2035                                      [Thriller]   \n",
       "2109                                [Drama, Romance]   \n",
       "2734                                         [Drama]   \n",
       "\n",
       "                         extracted_production_companies  \\\n",
       "152   [Twentieth Century Fox Film Corporation, Dream...   \n",
       "1770  [TriStar Pictures, Sony Pictures Releasing, Sm...   \n",
       "2035  [StudioCanal, Film4, Anton Capital Entertainme...   \n",
       "2109  [New Line Cinema, Sunswept Entertainment, Metr...   \n",
       "2734    [TriStar Pictures, Sony Pictures Entertainment]   \n",
       "\n",
       "         extracted_production_countries  \\\n",
       "152   [China, United States of America]   \n",
       "1770         [United States of America]   \n",
       "2035           [France, United Kingdom]   \n",
       "2109         [United States of America]   \n",
       "2734         [United States of America]   \n",
       "\n",
       "                                     extracted_keywords  \\\n",
       "152   [china, martial arts, kung fu, village, panda,...   \n",
       "1770  [bomb, sniper, tv show, hostage drama, police,...   \n",
       "2035                   [based on novel, woman director]   \n",
       "2109  [england, based on novel, depression, small to...   \n",
       "2734  [miracle, christian, cure, woman director, acc...   \n",
       "\n",
       "        extracted_spoken_languages  is_woman_director  release_year  \n",
       "152                      [English]                  1          2016  \n",
       "1770                     [English]                  1          2016  \n",
       "2035  [Pусский, English, Français]                  1          2016  \n",
       "2109                     [English]                  1          2016  \n",
       "2734                     [English]                  1          2016  \n",
       "\n",
       "[5 rows x 27 columns]"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# The first condition checks if the values in the 'is_woman_director' column are equal to 1, indicating a woman director.\n",
    "# The second condition checks if the values in the 'release_year' column are equal to 2016.\n",
    "# Both conditions must be met.\n",
    "df[(df.is_woman_director == 1) & (df.release_year == 2016)]\n",
    "#In this dataset, according to release date the last movie, directed by a woman was in 2016-06-12, "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dde4c817-190b-4879-acc0-689a462bb5a9",
   "metadata": {},
   "source": [
    "##### The last movie with a keyword woman-director is 'Me Before You' 2016, the main production company - New Line Cinema."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7c34ab5-ec3e-46fe-bb74-de64bee8a12a",
   "metadata": {},
   "source": [
    "##### To comapare the budgets of movies it fair to look at movies, produced by the same company, since the disperion between different companies' budgets would effect the results."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f72fc1c-8cb3-4b9b-8f5a-39289f30c1e3",
   "metadata": {},
   "source": [
    "### Finding all New Line Cinema/Productions movies from 2016"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 354,
   "id": "512fb85f-6f80-49ad-9e71-ee1b856ff681",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>budget</th>\n",
       "      <th>genres</th>\n",
       "      <th>homepage</th>\n",
       "      <th>id</th>\n",
       "      <th>keywords</th>\n",
       "      <th>original_language</th>\n",
       "      <th>original_title</th>\n",
       "      <th>overview</th>\n",
       "      <th>popularity</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>production_countries</th>\n",
       "      <th>release_date</th>\n",
       "      <th>revenue</th>\n",
       "      <th>runtime</th>\n",
       "      <th>spoken_languages</th>\n",
       "      <th>status</th>\n",
       "      <th>tagline</th>\n",
       "      <th>title</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>extracted_genres</th>\n",
       "      <th>extracted_production_companies</th>\n",
       "      <th>extracted_production_countries</th>\n",
       "      <th>extracted_keywords</th>\n",
       "      <th>extracted_spoken_languages</th>\n",
       "      <th>release_year</th>\n",
       "      <th>is_woman_director</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>914</th>\n",
       "      <td>50000000</td>\n",
       "      <td>[{\"id\": 28, \"name\": \"Action\"}, {\"id\": 35, \"nam...</td>\n",
       "      <td>http://www.centralintelligencemovie.com/</td>\n",
       "      <td>302699</td>\n",
       "      <td>[{\"id\": 470, \"name\": \"spy\"}, {\"id\": 591, \"name...</td>\n",
       "      <td>en</td>\n",
       "      <td>Central Intelligence</td>\n",
       "      <td>After he reunites with an old pal through Face...</td>\n",
       "      <td>45.318703</td>\n",
       "      <td>[{\"name\": \"New Line Cinema\", \"id\": 12}, {\"name...</td>\n",
       "      <td>[{\"iso_3166_1\": \"US\", \"name\": \"United States o...</td>\n",
       "      <td>2016-06-15</td>\n",
       "      <td>216972543</td>\n",
       "      <td>107.0</td>\n",
       "      <td>[{\"iso_639_1\": \"en\", \"name\": \"English\"}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>Saving the world takes a little Hart and a big...</td>\n",
       "      <td>Central Intelligence</td>\n",
       "      <td>6.2</td>\n",
       "      <td>1650</td>\n",
       "      <td>[Action, Comedy]</td>\n",
       "      <td>[New Line Cinema, Universal Pictures, Bluegras...</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[spy, cia, espionage, high school reunion, ref...</td>\n",
       "      <td>[English]</td>\n",
       "      <td>2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1160</th>\n",
       "      <td>40000000</td>\n",
       "      <td>[{\"id\": 27, \"name\": \"Horror\"}]</td>\n",
       "      <td>http://www.warnerbros.com/conjuring-2</td>\n",
       "      <td>259693</td>\n",
       "      <td>[{\"id\": 212, \"name\": \"london england\"}, {\"id\":...</td>\n",
       "      <td>en</td>\n",
       "      <td>The Conjuring 2</td>\n",
       "      <td>Lorraine and Ed Warren travel to north London ...</td>\n",
       "      <td>68.794673</td>\n",
       "      <td>[{\"name\": \"New Line Cinema\", \"id\": 12}, {\"name...</td>\n",
       "      <td>[{\"iso_3166_1\": \"CA\", \"name\": \"Canada\"}, {\"iso...</td>\n",
       "      <td>2016-05-13</td>\n",
       "      <td>320170008</td>\n",
       "      <td>134.0</td>\n",
       "      <td>[{\"iso_639_1\": \"en\", \"name\": \"English\"}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>The next true story from the case files of Ed ...</td>\n",
       "      <td>The Conjuring 2</td>\n",
       "      <td>7.0</td>\n",
       "      <td>1949</td>\n",
       "      <td>[Horror]</td>\n",
       "      <td>[New Line Cinema, Dune Entertainment, The Safr...</td>\n",
       "      <td>[Canada, United States of America]</td>\n",
       "      <td>[london england, england, 1970s, spirit, singl...</td>\n",
       "      <td>[English]</td>\n",
       "      <td>2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1335</th>\n",
       "      <td>38000000</td>\n",
       "      <td>[{\"id\": 35, \"name\": \"Comedy\"}, {\"id\": 10749, \"...</td>\n",
       "      <td>http://howtobesinglemovie.com/</td>\n",
       "      <td>259694</td>\n",
       "      <td>[{\"id\": 242, \"name\": \"new york\"}, {\"id\": 818, ...</td>\n",
       "      <td>en</td>\n",
       "      <td>How to Be Single</td>\n",
       "      <td>New York City is full of lonely hearts seeking...</td>\n",
       "      <td>46.078371</td>\n",
       "      <td>[{\"name\": \"New Line Cinema\", \"id\": 12}, {\"name...</td>\n",
       "      <td>[{\"iso_3166_1\": \"US\", \"name\": \"United States o...</td>\n",
       "      <td>2016-01-21</td>\n",
       "      <td>112343513</td>\n",
       "      <td>110.0</td>\n",
       "      <td>[{\"iso_639_1\": \"en\", \"name\": \"English\"}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>Welcome to the party</td>\n",
       "      <td>How to Be Single</td>\n",
       "      <td>5.9</td>\n",
       "      <td>1201</td>\n",
       "      <td>[Comedy, Romance]</td>\n",
       "      <td>[New Line Cinema, Flower Films, Metro-Goldwyn-...</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[new york, based on novel, one-night stand, si...</td>\n",
       "      <td>[English]</td>\n",
       "      <td>2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2579</th>\n",
       "      <td>15000000</td>\n",
       "      <td>[{\"id\": 28, \"name\": \"Action\"}, {\"id\": 35, \"nam...</td>\n",
       "      <td>http://keanumovie.com/</td>\n",
       "      <td>342521</td>\n",
       "      <td>[{\"id\": 2708, \"name\": \"hitman\"}, {\"id\": 3688, ...</td>\n",
       "      <td>en</td>\n",
       "      <td>Keanu</td>\n",
       "      <td>Friends hatch a plot to retrieve a stolen cat ...</td>\n",
       "      <td>18.268009</td>\n",
       "      <td>[{\"name\": \"New Line Cinema\", \"id\": 12}]</td>\n",
       "      <td>[{\"iso_3166_1\": \"US\", \"name\": \"United States o...</td>\n",
       "      <td>2016-04-21</td>\n",
       "      <td>20566327</td>\n",
       "      <td>94.0</td>\n",
       "      <td>[{\"iso_639_1\": \"en\", \"name\": \"English\"}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>Kitten, please.</td>\n",
       "      <td>Keanu</td>\n",
       "      <td>6.0</td>\n",
       "      <td>421</td>\n",
       "      <td>[Action, Comedy]</td>\n",
       "      <td>[New Line Cinema]</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[hitman, strip club, african american, gangste...</td>\n",
       "      <td>[English]</td>\n",
       "      <td>2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3558</th>\n",
       "      <td>4900000</td>\n",
       "      <td>[{\"id\": 27, \"name\": \"Horror\"}, {\"id\": 53, \"nam...</td>\n",
       "      <td>http://www.lightsoutmovie.com/</td>\n",
       "      <td>345911</td>\n",
       "      <td>[{\"id\": 236, \"name\": \"suicide\"}, {\"id\": 3762, ...</td>\n",
       "      <td>en</td>\n",
       "      <td>Lights Out</td>\n",
       "      <td>When Rebecca left home, she thought she left h...</td>\n",
       "      <td>48.170508</td>\n",
       "      <td>[{\"name\": \"New Line Productions\", \"id\": 8781},...</td>\n",
       "      <td>[{\"iso_3166_1\": \"US\", \"name\": \"United States o...</td>\n",
       "      <td>2016-07-21</td>\n",
       "      <td>44107032</td>\n",
       "      <td>81.0</td>\n",
       "      <td>[{\"iso_639_1\": \"en\", \"name\": \"English\"}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Lights Out</td>\n",
       "      <td>6.3</td>\n",
       "      <td>1129</td>\n",
       "      <td>[Horror, Thriller]</td>\n",
       "      <td>[New Line Productions, Matter Productions, Ato...</td>\n",
       "      <td>[United States of America]</td>\n",
       "      <td>[suicide, darkness, basement, based on short s...</td>\n",
       "      <td>[English]</td>\n",
       "      <td>2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        budget                                             genres  \\\n",
       "914   50000000  [{\"id\": 28, \"name\": \"Action\"}, {\"id\": 35, \"nam...   \n",
       "1160  40000000                     [{\"id\": 27, \"name\": \"Horror\"}]   \n",
       "1335  38000000  [{\"id\": 35, \"name\": \"Comedy\"}, {\"id\": 10749, \"...   \n",
       "2579  15000000  [{\"id\": 28, \"name\": \"Action\"}, {\"id\": 35, \"nam...   \n",
       "3558   4900000  [{\"id\": 27, \"name\": \"Horror\"}, {\"id\": 53, \"nam...   \n",
       "\n",
       "                                      homepage      id  \\\n",
       "914   http://www.centralintelligencemovie.com/  302699   \n",
       "1160     http://www.warnerbros.com/conjuring-2  259693   \n",
       "1335            http://howtobesinglemovie.com/  259694   \n",
       "2579                    http://keanumovie.com/  342521   \n",
       "3558            http://www.lightsoutmovie.com/  345911   \n",
       "\n",
       "                                               keywords original_language  \\\n",
       "914   [{\"id\": 470, \"name\": \"spy\"}, {\"id\": 591, \"name...                en   \n",
       "1160  [{\"id\": 212, \"name\": \"london england\"}, {\"id\":...                en   \n",
       "1335  [{\"id\": 242, \"name\": \"new york\"}, {\"id\": 818, ...                en   \n",
       "2579  [{\"id\": 2708, \"name\": \"hitman\"}, {\"id\": 3688, ...                en   \n",
       "3558  [{\"id\": 236, \"name\": \"suicide\"}, {\"id\": 3762, ...                en   \n",
       "\n",
       "            original_title                                           overview  \\\n",
       "914   Central Intelligence  After he reunites with an old pal through Face...   \n",
       "1160       The Conjuring 2  Lorraine and Ed Warren travel to north London ...   \n",
       "1335      How to Be Single  New York City is full of lonely hearts seeking...   \n",
       "2579                 Keanu  Friends hatch a plot to retrieve a stolen cat ...   \n",
       "3558            Lights Out  When Rebecca left home, she thought she left h...   \n",
       "\n",
       "      popularity                               production_companies  \\\n",
       "914    45.318703  [{\"name\": \"New Line Cinema\", \"id\": 12}, {\"name...   \n",
       "1160   68.794673  [{\"name\": \"New Line Cinema\", \"id\": 12}, {\"name...   \n",
       "1335   46.078371  [{\"name\": \"New Line Cinema\", \"id\": 12}, {\"name...   \n",
       "2579   18.268009            [{\"name\": \"New Line Cinema\", \"id\": 12}]   \n",
       "3558   48.170508  [{\"name\": \"New Line Productions\", \"id\": 8781},...   \n",
       "\n",
       "                                   production_countries release_date  \\\n",
       "914   [{\"iso_3166_1\": \"US\", \"name\": \"United States o...   2016-06-15   \n",
       "1160  [{\"iso_3166_1\": \"CA\", \"name\": \"Canada\"}, {\"iso...   2016-05-13   \n",
       "1335  [{\"iso_3166_1\": \"US\", \"name\": \"United States o...   2016-01-21   \n",
       "2579  [{\"iso_3166_1\": \"US\", \"name\": \"United States o...   2016-04-21   \n",
       "3558  [{\"iso_3166_1\": \"US\", \"name\": \"United States o...   2016-07-21   \n",
       "\n",
       "        revenue  runtime                          spoken_languages    status  \\\n",
       "914   216972543    107.0  [{\"iso_639_1\": \"en\", \"name\": \"English\"}]  Released   \n",
       "1160  320170008    134.0  [{\"iso_639_1\": \"en\", \"name\": \"English\"}]  Released   \n",
       "1335  112343513    110.0  [{\"iso_639_1\": \"en\", \"name\": \"English\"}]  Released   \n",
       "2579   20566327     94.0  [{\"iso_639_1\": \"en\", \"name\": \"English\"}]  Released   \n",
       "3558   44107032     81.0  [{\"iso_639_1\": \"en\", \"name\": \"English\"}]  Released   \n",
       "\n",
       "                                                tagline                 title  \\\n",
       "914   Saving the world takes a little Hart and a big...  Central Intelligence   \n",
       "1160  The next true story from the case files of Ed ...       The Conjuring 2   \n",
       "1335                               Welcome to the party      How to Be Single   \n",
       "2579                                    Kitten, please.                 Keanu   \n",
       "3558                                                NaN            Lights Out   \n",
       "\n",
       "      vote_average  vote_count    extracted_genres  \\\n",
       "914            6.2        1650    [Action, Comedy]   \n",
       "1160           7.0        1949            [Horror]   \n",
       "1335           5.9        1201   [Comedy, Romance]   \n",
       "2579           6.0         421    [Action, Comedy]   \n",
       "3558           6.3        1129  [Horror, Thriller]   \n",
       "\n",
       "                         extracted_production_companies  \\\n",
       "914   [New Line Cinema, Universal Pictures, Bluegras...   \n",
       "1160  [New Line Cinema, Dune Entertainment, The Safr...   \n",
       "1335  [New Line Cinema, Flower Films, Metro-Goldwyn-...   \n",
       "2579                                  [New Line Cinema]   \n",
       "3558  [New Line Productions, Matter Productions, Ato...   \n",
       "\n",
       "          extracted_production_countries  \\\n",
       "914           [United States of America]   \n",
       "1160  [Canada, United States of America]   \n",
       "1335          [United States of America]   \n",
       "2579          [United States of America]   \n",
       "3558          [United States of America]   \n",
       "\n",
       "                                     extracted_keywords  \\\n",
       "914   [spy, cia, espionage, high school reunion, ref...   \n",
       "1160  [london england, england, 1970s, spirit, singl...   \n",
       "1335  [new york, based on novel, one-night stand, si...   \n",
       "2579  [hitman, strip club, african american, gangste...   \n",
       "3558  [suicide, darkness, basement, based on short s...   \n",
       "\n",
       "     extracted_spoken_languages  release_year  is_woman_director  \n",
       "914                   [English]          2016                  0  \n",
       "1160                  [English]          2016                  0  \n",
       "1335                  [English]          2016                  0  \n",
       "2579                  [English]          2016                  0  \n",
       "3558                  [English]          2016                  0  "
      ]
     },
     "execution_count": 354,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Filter the DataFrame 'df' for rows where: 'release_year' is 2016, 'is_woman_director' is 0 (no woman director),\n",
    "# 'extracted_production_companies' contains 'New Line' in any of the company names.\n",
    "\n",
    "# The resulting DataFrame includes movies from 2016 without a woman director, but with 'New Line' in the production company names.\n",
    "df[(df.release_year == 2016) & (df.is_woman_director == 0) & (df['extracted_production_companies'].apply(lambda x: any('New Line' in company for company in x)))]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a7a8ee48-a53c-4ac3-985a-c4d4af3f58ca",
   "metadata": {},
   "source": [
    "##### There are 6 movies in total, produced by New Line Cinema/Productions in 2016. I further investigate their budgets and revenue."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "05062856-5307-4a64-bf08-2df8318ac77f",
   "metadata": {},
   "source": [
    "### Comapring the budget and revenue of woman and non-woman directed New Line Cinema/Productions movies"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7a9f6cb-0fe3-41a2-96de-9ab85cc04128",
   "metadata": {},
   "source": [
    "##### At first I include all 6 movies in a new dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "f7ae5e9d-6761-44bb-8624-cf87b73b33d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_plot1 = df[['original_title', 'budget', 'revenue']][(df.id == 345911) | (df.id == 342521) | (df.id == 259694) | (df.id == 259693) | (df.id == 302699) | (df.id == 296096)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "45b24068-d6d3-45db-9457-eaa0a44c12ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reshape the DataFrame 'df_plot1' using the 'pd.melt' function.\n",
    "# The 'id_vars' parameter is set to \"original_title\", indicating that the column \"original_title\" will be kept as an identifier variable.\n",
    "# The resulting DataFrame 'df_plot11' has a \"variable\" column containing the column names of 'df_plot1' and a \"value\" column containing the corresponding values.\n",
    "\n",
    "# Sort the reshaped DataFrame 'df_plot11' by the \"value\" column in ascending order.\n",
    "df_plot11 = pd.melt(df_plot1, id_vars = \"original_title\").sort_values(by='value', ascending=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2f033f29-108d-4a0c-9eab-a31e73a51d21",
   "metadata": {},
   "source": [
    "##### Chaning the column names for better readability."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "2e9a761b-c1cf-41f7-ab91-8e7187cba0d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Adjust column names\n",
    "df_plot11.rename(columns = {'original_title' : 'Title', 'variable' : 'Budget/Revenue', 'value' : 'Hundred Millions USD'}, inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d2c669e-ae37-4409-9526-532361f55304",
   "metadata": {},
   "source": [
    "##### Plotting the budget and revenue of all New Lines Cinema/Productions movies from 2016."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "38842472-9362-4c2d-9bb2-e79eaba3e4d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 442.5x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create a categorical plot using seaborn's catplot.\n",
    "sns.set_style('whitegrid')\n",
    "plot = sns.catplot(x= 'Title',# Set the x-axis as 'Title'\n",
    "                   y='Hundred Millions USD', # Set the y-axis as 'Hundred Millions USD' (budget)\n",
    "                   hue = 'Budget/Revenue', # Set the hue as 'Budget/Revenue'.\n",
    "                   data=df_plot11, # The data is taken from the DataFrame 'df_plot11'.\n",
    "                   kind='bar', # Specify the kind of plot as 'bar'.\n",
    "                   palette=['pink', 'purple']) # Set the palette parameter to a list of colors\n",
    " \n",
    "# Set the title\n",
    "plt.title('Budget/Revenue of New Line movies in 2016')\n",
    "\n",
    "# Rotate x-axis labels\n",
    "plot.set_xticklabels(rotation=45)\n",
    "\n",
    "# Show the plot\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8015c70-278c-4238-9611-fe03cca0213f",
   "metadata": {},
   "source": [
    "##### From the graph 'Me Before you\" had the 3rd lowest budget out of all New Line movies in 2016. More than twice lower than the highest budget movie - Central Intelligence. Interestingly, their revenue is almost the same."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a7f3fbbb-67e9-4c89-951c-978dcdc7ee00",
   "metadata": {},
   "source": [
    "### Conclusion \n",
    "##### The answer to the questions 'Do woman-directed movies receive less budget? What about revenue?' cannot be generalized based on these findings, since again it might be the case that not all woman-directed movies had a keyword of a kind and there might be some other underlying factors, determining the budget.\n",
    "\n",
    "##### However, in this dataset the latest woman-directed New Line movie received more than twice (40%) lower budget, compared to the highest budget New Line movie in 2016. Despite the lower budget, it managed to produce almost the same revenue. \n",
    "##### Studies on gender dispension in movie industry suggest that woman directors tend to receive 63% less budget between years of 2010 and 2015. It is evident, that women are not included and trusted enougth to receive fair budgets, despice them beinging capable to generate comepetitive revenue.\n",
    "\n",
    "[Source](https://www.hollywoodreporter.com/news/general-news/study-films-directed-by-women-907229/)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
