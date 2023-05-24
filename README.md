# The Supreme Court Of The United States (SCOTUS)

<img src="images/scotus_seal.png" alt="scotus image" style="display: block; margin-left: auto; margin-right: auto"/>

Understanding the judicial decision-making process may be aided by the ability to predict the results of legal matters. In light of this, we trained a model that can predict and classify the court's decision given the case's information from the plaintiff and the defendant in textual format. This was achieved by using powerful NLP algorithms to evaluate past court cases. As a result, by producing a verdict, the model is simulating a human jury. 

Overall, this solution can be utilized to speed up judicial decision-making and lighten the load on judges.

# Libraries used

For the project the following Python libraries for Data Science and Machine Learning were used:
| Package | Function |
|---------|----------|
| numpy | For scientific computation |
| pandas | For data manipulation |
| matplotlib | For visualization |
| seaborn | For visualization |
| tensorflow | For building neural networks |
| nltk | For text preprocessing |
| sklearn | For machine learning |
| wordcloud | For creating wordclouds |
| spacy | For lemmatization |
| streamlit | For application development |

# Methodology

<img src="images/methodology.jpg" alt="framework" style="display: block; margin-left: auto; margin-right: auto"/>

# About the dataset

### Source

The dataset used was obtained from [Kaggle](https://www.kaggle.com/datasets/deepcontractor/supreme-court-judgment-prediction).
The dataset contains 3304 cases from the Supreme Court of the United States from 1955 to 2021. Each case has the case's identifiers as well as the facts of the case and the decision outcome.

Target Variable: First_Party_Winner, if true means that the first party won, and if false it means that the second party won.

### Features

`ID`

`name`

`href`

`docket`

`term`

`first_party`

`second_part`

`facts`

`facts_len`

`majority_vote`

`minority_vote`

`first_party_winner`

`decision_type`

`disposition`

`issue_area`

# Modeling (Random Forest, Extreme Gradient Boosting, LSTM)

- Random Forest:

- Extreme Gradient Boosting:

- LSTM
