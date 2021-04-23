# Job_Recommendation

## Data Set

There are 4 datasets available on kaggle.

* The Combined_Jobs_Final.csv file: has the main jobs data(title, description, company, etc.).
* The Job_Views.csv file: the file with the jobs seeing for the user.
* The Experience.csv: the file containing the experience from the user.
* The Positions_Of_Interest.csv: contains the interest the user previously has manifested.

## Approach
1. Merged all the 4 datasets based on Applicant_ID and selected useful columns.Then, imputed missing values,removed stop words, performed lemmatization and merged the columns to create corpus.
2. Extracted features using TF-IDF and count vectorizer.
3. Built content based recommendor using Spacy.

## Sample Results

![Capture](https://user-images.githubusercontent.com/30667531/115827233-a237f280-a3d1-11eb-93d2-1d07c3c7214f.PNG)

## References

https://medium.com/@chaitanyarb619/recommendation-systems-a-walk-trough-33587fecc195 <br />
https://www.kaggle.com/c/job-recommendation <br />
https://medium.com/@armandj.olivares/building-nlp-content-based-recommender-systems-b104a709c042 <br />
