# **Comparison of methods for extracting, retrieving and ranking coherent phrases in long posts from debate portals**

## **General**
This script queries 5933 argumentative texts relating to predefined topics and queries. We chose the Jaccard Coefficient, BM25, BERT, TF-IDF and ChatGPT to match the query with the arguments. These are stored in a JSON which will be evaluated afterwards.

## **Setup**
1. Install all required packages with: ```pip install -r requirements.txt```
2. Create a ChatGPT account if you don't have one yet.
2. Create a ```access_token.json``` file with the access token of your ChatGPT session. You can get your token from https://chat.openai.com/api/auth/session.

    ```
    {
        "access_token": "your_access_token"
    }
    ```
3. 

