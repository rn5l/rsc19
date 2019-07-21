# rsc19
Publication of the code we used in the RecSys Challenge 2018.

## Requirements
* The code was tested with miniconda3 (python3.7.3)
* All neccessary libraries can be installed via requirements_(conda|pip).txt
```console
conda install --yes --file requirements_conda.txt
pip install -r requirements_pip.txt
```

## Preprocessing
* Extract the provided data zip file to the folder *data/raw*
* First, run the preprocessing script for the item metadata:
  * ```console
    python -m preprocess.extend_meta
    ```
* Second, transform the log data:
  * ```console
    python -m preprocess.transform
    ```
* Third, split the log data:
  * ```console
    python -m preprocess.split
    ```
* The data will now have been preprocessed to the folder *data/competition/*

## Crawling
* In addition to the existing item metadata, we crawled some additional information
  * As the crawling might take a few days, the results are included in the *data/crawled* folder. 
* First, run the following script to crawl more item information from trivago:
  * ```console
    python -m crawl.crawl_item_info
    ```
* Second, run the following scripts to crawl coordinates for cities and pois from google maps:
  * ```console
    python -m crawl.crawl_city_info
    ```
  * ```console
    python -m crawl.crawl_poi_info
    ```
  * To run these scripts the Google Maps API-Key has to be set in the file. 


## Creating Latent Factors
* As a part of our final model we created a few features based on latent representations of items and sessions
* First, run the following script to create item doc2vec representations from the metadata
  * ```console
    python -m latent.meta_w2v
    ```
* Second, run the following script to create item and session doc2vec representations from the log data
  * ```console
    python -m latent.session_w2v
    ```
* Third, create item and session representations from the log data using BPR
  * ```console
    python -m latent.session_nmf
    ```

## Creating the Feature Set and Training the Model
* With unlimited memory the feature set can be constructed and the model can be trained with a single call to:
  * ```console
    python lgbm_cv.py
    ```
* To save memory the parts of the feature set can be constructed individually
  * ```console
    python -m featureset.*
    ```
  * The parts partially rely on eachother and should be created in the following order:
    * popularity
    * price
    * meta
    * crawl
    * time
    * properties
    * latent_sim
    * geo
    * session
    * user
    * list_context
    * position
    * stars
    * rank
    * combined
  * Afterwards, the following script can be used to train the model in a memory-saving fashion:
    * ```console
      python lgbm_cv_file.py
      ```

## Building the Submission
* Run the following python script to convert our internal solution format with confidence values to the official format
  * ```console
    python submission.py
    ```
