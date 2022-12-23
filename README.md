# Quora - Duplicate Question Detection
Dataset can be found [here](https://www.kaggle.com/c/quora-question-pairs/data)  
300-dimensional GloVe embeddings can be found [here](https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt)

## Linear Models
### Logistic Regression

|Feature Set|Accuracy|F1-score|
|:---:|:---:|:---:|
|Unigram|74.18|63.11|
|Unigram + Bigram|79.62|70.66|
|Unigram + Bigram + Trigram|81.15|71.48|

### Support Vector Machine
|Model|Accuracy|F1-score|
|:---:|:---:|:---:|
|Linear SVM, Unigram|73.39|64.13|
|Linear SVM, Bigram|77.65|69.94|
|Linear SVM, Trigram|79.26|71.32|

After fine tuning the hyperparameters, the best model is Linear SVM with Trigram features with an accuracy of 80.11% and F1-score of 71.29%.

### SVM using sentence embdeddings
We used plain sentence embeddings and distance based metrics. RBF kernel seemed to perform better than linear kernel.

For plain sentence embeddings:

|Model|Accuracy|F1-score|
|:---:|:---:|:---:|
|Linear SVM|63.85|61.93|
|RBF SVM|77.38|69.89|

For distance based features:

|Model|Accuracy|F1-score|
|:---:|:---:|:---:|
|Linear SVM|63.91|62.53|
|RBF SVM|67.94|68.34|

## Tree-based Models
We used decision trees, random forests and gradient boosted trees. We took a prefix of 7 feature sets and used them as input to the models. The final accuracy in the table below is the average of the 3 model accuracies (approximately same for each model).

|Feature Set|Num-Features|Accuracy|F1-score|
|:---:|:---:|:---:|:---:|
|L|4|64.32|23.28|
|L,LC|6|69.61|58.35|
|L,LC,LCXS|8|69.60|58.24|
|L,LC,LCXS,LW|9|72.33|62.01|
|L,LC,LCXS,LW,CAP|11|72.34|62.11|
|L,LC,LCXS,LW,CAP,PRE|19|72.54|62.93|
|L,LC,LCXS,LW,CAP,PRE,M|25|74.01|65.13|

## Neural Networks
Four models, viz Continuous Bag of Words (CBOW), Long Short Term Memory (LSTM), Bidirectional LSTM (BiLSTM) and LSTM with Attention (LSTM-Att) were used. 

|Model|Accuracy|F1-score|
|:---:|:---:|:---:|
|CBOW|80.65|83.36|
|LSTM|78.41|73.57|
|BiLSTM|79.09|76.37|
|LSTM-Att|77.51|70.64|