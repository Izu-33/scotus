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

| :Column: | Description |
|--------|-------------|
| `ID` | Unique case identiﬁer |
| `name` | The name of the case |
| `href` | The Oyez’s API URL for the case |
| `docket` | A special identiﬁer of the case used by the legal system |
| `term` | The year when the Court received the case |
| `first_party` | The name of the ﬁrst party (petitioner) |
| `second_part` | The name of the second party (respondent) |
| `facts` | The absolute, neutral facts of the case written by the court clerk |
| `facts_len` | The number of justices voting for the majority opinion |
| `majority_vote` | The number of justices voting for the majority opinion |
| `minority_vote` | The number of justices voting for the minority opinion |
| `first_party_winner` | True if the ﬁrst party won the case, otherwise False and the second party won the case |
| `decision_type` | The type of the decision decided by the court, e.g.: per curiam, equally divided,opinion of the court |
| `disposition` | The treatment the Supreme Court accorded the court whose decision it reviewed;e.g.: afﬁrmed, reversed, vacated |
| `issue_area` | The pre-deﬁned legal issue category of the case; e.g.: Civil Rights, CriminalProcedure, Federal Taxation |

# Modeling (Random Forest, Extreme Gradient Boosting, LSTM)

- Random Forest:

- Extreme Gradient Boosting:

- LSTM
