# 🌍 African Medical Data Fetcher Explanation 🌍

Hey future Python rockstar! 🚀 This document explains the `african_medical_fetcher_fixed.py` script step by step. This script is designed to collect African medical research data from PubMed (a huge database of medical research papers). Let's break it down into digestible chunks!

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Imports and Setup](#imports-and-setup)
3. [The Main Class](#the-main-class)
4. [Database Setup](#database-setup)
5. [Query Generation](#query-generation)
6. [Data Fetching](#data-fetching)
7. [Data Processing](#data-processing)
8. [Saving Results](#saving-results)
9. [Running the Script](#running-the-script)

## Introduction

This script is a specialized tool designed to fetch medical research papers related to African health topics from PubMed. It focuses on:

- Searching for papers about specific African countries and health issues
- Filtering for research that's actually relevant to Africa
- Saving the results in a structured format
- Using error handling to make the code resilient

The script uses asyncio (for asynchronous programming) to make multiple requests efficiently!

## Imports and Setup

```python
import asyncio
import aiohttp
import requests
import time
import json
import os
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Set
import xml.etree.ElementTree as ET
```

**What's happening here?** 🤔

- `asyncio` and `aiohttp`: Used for asynchronous operations (like making multiple web requests at once)
- `requests`: A library for making HTTP requests
- `time`: For adding delays between requests
- `json`: For working with JSON data
- `os`: For interacting with the operating system (like creating directories)
- `sqlite3`: For working with a SQLite database
- `hashlib`: For creating hash values (unique IDs)
- `datetime`: For working with dates and times
- `typing`: For adding type hints to the code
- `xml.etree.ElementTree`: For parsing XML data from PubMed

## The Main Class

The core of the script is the `AfricanMedicalFetcher` class. Think of this as a specialized robot 🤖 designed to find and collect African medical research!

### Initialization

The `__init__` method sets up the fetcher with:

```python
def __init__(self):
    self.email = "eruwagolden55@gmail.com"  # Used in API requests
    self.delay = 0.34  # Delay between requests (to be polite to PubMed)
    
    # Database setup
    self.db_path = "/Users/new/elara_main/data/elara_cache.db"
    self.setup_african_cache()
    
    # Lists of African countries, health priorities, and more...
```

**What's happening here?** 🤔

- Sets an email for PubMed API requests (they ask for this to track usage)
- Sets a small delay between requests (0.34 seconds) to avoid overwhelming PubMed's servers
- Defines the path to a SQLite database for caching results
- Calls `setup_african_cache()` to initialize the database
- Creates lists of African countries, health priorities, specialties, and research terms

### Database Setup

```python
def setup_african_cache(self):
    """Setup African-specific cache tables"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # African articles cache
    cursor.execute('''CREATE TABLE IF NOT EXISTS african_articles (...)''')
    
    # African query cache
    cursor.execute('''CREATE TABLE IF NOT EXISTS african_queries (...)''')
    
    conn.commit()
    conn.close()
```

**What's happening here?** 🤔

- Connects to a SQLite database
- Creates two tables if they don't exist:
  - `african_articles`: Stores information about fetched articles
  - `african_queries`: Keeps track of which queries have been executed
- This is like creating a filing cabinet 🗄️ for organizing the research papers

### Loading Cache

```python
def load_african_cache(self):
    """Load existing African medical data cache"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Load cached African PMIDs
    cursor.execute("SELECT pmid FROM african_articles")
    self.african_pmids = {row[0] for row in cursor.fetchall()}
    
    # Load completed African queries
    cursor.execute("SELECT query_hash FROM african_queries")
    self.completed_african_queries = {row[0] for row in cursor.fetchall()}
```

**What's happening here?** 🤔

- Connects to the SQLite database again
- Loads two sets of information:
  - `african_pmids`: IDs of articles already fetched (to avoid duplicates)
  - `completed_african_queries`: Queries already run (to avoid repeating work)
- This is like checking what's already in your filing cabinet before starting work 🔍

## Query Generation

```python
def generate_african_queries(self) -> List[Dict]:
    """Generate comprehensive African medical research queries"""
    queries = []
    
    # 1. Country-specific health research
    for country in self.african_countries:
        queries.append({
            'query': country + ' health',
            'type': 'country_health',
            'focus': country
        })
    # ... more query generation ...
```

**What's happening here?** 🤔

- Creates a list of search queries for PubMed
- Each query is a dictionary with:
  - `query`: The actual search term to send to PubMed
  - `type`: A category for the query
  - `focus`: What the query is focusing on
- The method creates queries based on:
  - Country-specific health research
  - Endemic diseases in Africa
  - Medical specialties in Africa
  - Regional research terms
  - Country + Disease combinations
  - African traditional medicine
  - African health policy and systems
- This is like making a shopping list 🛒 of all the information we want to find!

## Data Fetching

The data fetching process happens in several methods, but it starts with the main collection method:

```python
async def african_data_collection(self, max_articles_per_query: int = 200):
    """AFRICAN-FOCUSED data collection with robust error handling!"""
    queries = self.generate_african_queries()
    
    # Filter new queries
    new_queries = [q for q in queries if not self.is_african_query_completed(q)]
```

**What's happening here?** 🤔

- Gets the list of queries from `generate_african_queries()`
- Filters out queries that have already been completed
- For each new query, it will:
  1. Search PubMed using `_search_pubmed_async()`
  2. Filter out already fetched articles
  3. Fetch article details using `_fetch_articles_async()`
  4. Filter by African relevance
  5. Save the results
  6. Mark the query as completed
  
The actual requests to PubMed are made using these helper methods:

### `_search_pubmed_async`

```python
async def _search_pubmed_async(self, query: str, max_results: int) -> List[str]:
    """Async PubMed search with timeout handling"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'json',
        'email': self.email,
        'tool': 'ElaraAI_AfricanFetcher'
    }
```

**What's happening here?** 🤔

- Makes an asynchronous request to PubMed's search API
- Returns a list of article IDs (PMIDs) that match the query
- Includes error handling for timeouts and other issues

### `_fetch_articles_async` and `_fetch_batch_async`

These methods fetch detailed information about the articles:

```python
async def _fetch_articles_async(self, pmids: List[str]) -> List[Dict]:
    """Fetch articles with robust error handling"""
    articles = []
    batch_size = 10  # Smaller batches for more stability
    
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        try:
            batch_articles = await self._fetch_batch_async(batch)
            # ...
```

**What's happening here?** 🤔

- Takes a list of article IDs and fetches detailed information
- Processes the articles in small batches (10 at a time)
- Uses `_fetch_batch_async` to make the actual API requests
- Includes error handling to skip problematic batches

## Data Processing

Once the data is fetched, it needs to be processed. The key method is `_parse_african_xml`:

```python
def _parse_african_xml(self, xml_content: str) -> List[Dict]:
    """Parse XML with bulletproof error handling"""
    articles = []
    
    try:
        # Clean up XML content before parsing
        xml_content = xml_content.strip()
        if not xml_content:
            return articles
        
        root = ET.fromstring(xml_content)
        
        for article_elem in root.findall('.//PubmedArticle'):
            try:
                # Extract article information...
```

**What's happening here?** 🤔

- Takes XML content returned from PubMed and extracts useful information
- Uses ElementTree to parse the XML structure
- For each article element in the XML, it extracts:
  - Basic info (PMID, title, abstract)
  - Journal and publication date
  - Authors (with special attention to African authors)
  - Keywords and MeSH terms
- Creates a dictionary for each article with all this information
- Includes extensive error handling to skip problematic articles

### Calculating African Relevance

A key feature is calculating how relevant each article is to African health:

```python
def calculate_african_relevance(self, article: Dict) -> int:
    """Calculate African relevance score for an article"""
    score = 0
    text_to_check = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
    
    # Country mentions (high value)
    for country in self.african_countries:
        if country.lower() in text_to_check:
            score += 25
```

**What's happening here?** 🤔

- Calculates a score from 0-100 for how relevant an article is to African health
- Checks for mentions of:
  - African countries (25 points each)
  - Regional terms like "sub-saharan" (15 points each)
  - Endemic diseases (20 points each)
  - Research quality indicators (10 points each)
- This helps filter out articles that mention Africa but aren't really focused on it

## Saving Results

The results are saved in two ways:

1. In the SQLite database (for tracking purposes)
2. As JSON files (for data analysis)

```python
async def _save_african_batch(self, articles: List[Dict], query_info: Dict, batch_num: int) -> str:
    """Save African batch with enhanced metadata and error handling"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"/Users/new/elara_main/data/raw/african_batch_{batch_num:03d}_{query_info['type']}_{timestamp}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
```

**What's happening here?** 🤔

- Creates a filename based on the batch number, query type, and timestamp
- Makes sure the directory exists
- Calculates statistics about the batch:
  - Average African relevance score
  - Countries mentioned
  - Diseases mentioned
- Creates a JSON structure with batch info and article data
- Saves it to a file
- Returns the filename (to be stored in the database)

## Running the Script

The script can be run directly with the `run_african_collection` function:

```python
async def run_african_collection():
    """Execute African medical data collection with robust error handling!"""
    print("🌍" * 60)
    print("   AFRICAN MEDICAL DATA SUPER-FETCHER (FIXED VERSION)!")
    print("🌍" * 60)
    
    fetcher = AfricanMedicalFetcher()
    
    # Run collection (targeting ~200 articles per query for quality)
    new_articles = await fetcher.african_data_collection(max_articles_per_query=200)
```

**What's happening here?** 🤔

- Prints a fancy banner with lots of 🌍 emojis
- Creates an instance of the `AfricanMedicalFetcher` class
- Runs the collection process, targeting 200 articles per query
- Returns the list of new articles

And finally, the script has this block at the bottom:

```python
if __name__ == "__main__":
    # Launch the fixed African collection!
    asyncio.run(run_african_collection())
```

**What's happening here?** 🤔

- This is a special Python pattern that runs code only when the script is executed directly
- It calls `asyncio.run()` to run the asynchronous `run_african_collection()` function

## 🎮 Quiz Time! Test Your Understanding

1. What is the purpose of the `self.delay` variable?
   - A) To slow down the processing of data
   - B) To add a pause between API requests
   - C) To set the timeout for API requests

2. What does the `calculate_african_relevance` function do?
   - A) Checks if the authors are from Africa
   - B) Calculates how much an article focuses on African health topics
   - C) Determines if the article should be fetched

3. Why does the script use `asyncio` and `aiohttp`?
   - A) To make the code more complex
   - B) To handle multiple requests at the same time efficiently
   - C) Because PubMed requires it

4. What would happen if the script didn't use error handling (try/except blocks)?
   - A) It would run faster
   - B) It would crash if any single article had issues
   - C) It would collect more articles

**Answers:** 1:B, 2:B, 3:B, 4:B

## 🚀 Next Steps

Now that you understand this script, here are some fun things you could try:

1. Modify the script to search for a different region or medical topic
2. Add additional error handling or logging
3. Create a simple visualization of the collected data
4. Improve the relevance scoring algorithm

Happy coding! 🎉
