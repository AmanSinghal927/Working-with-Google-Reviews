We can scrape Google Reviews using 1. Scraping Google Reviews.py. Since some of the reviews could be code-switched hence, we can use Google 
Translate on these reviews using 2. Google Translation for Non English Reviews. Before beginning any text processing we must handle any
data inconsistencies which may have ocurred during data pull or tranlation using 3. Scraped DataFrame processing.py. The reviews are removed 
of any stop-words and numbers in 4. Cleaning the Reviews, and stemmed, lemmatized, POS tagged for creting two separate TF-IDF based word corpa
based on any classification flag in 5. TF-IDF Random Forest. The Random Forest Model which is used is also tuned by Hyper Parameter tuning
in 6. Hyper Parameter Tuning. 
