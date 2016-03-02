Create histograms for categorical variables and group/cluster them.

Use Vowpal Wabbit (vw-varinfo) or XGBoost (XGBfi) to quickly check two-way and three-way interactions.

Use statistical tests to discard noisy features (eg: select k best with ANOVA F-score), Benford's Law to detect natural counts (great for logtransforms).

Manually inspect the data and combine features that look similar in structure (both columns contain hashed variables) or expand categorical variables that look like hierarchical codes ("1.11125A" -> 1, 111, 25, A).

Use (progressive/cross-)validation with fast algorithms or on a subset of the data to spot significant changes.

Compute stats about a row of data (nr. of 0's, nr. of NAs, max, min, mean, std)

Transforms: Log, tfidf

Numerical->Categorical: Encoding numerical variables, like 11.25 as categorical variables: eleventwentyfive

Bayesian: Encode categorical variables with its ratio of the target variable in train set.

Reduce the dimensionality / project down with tSNE, MDA, PCA, LDA, RP, RBM, kmeans or expand raw data.

Genetic programming: http://gplearn.readthedocs.org/en/latest/examples.html#example-2-symbolic-tranformer to automatically create non-linear features.

Recursive Feature Elimination: Use all features -> drop the feature that results in biggest CV gain -> repeat till no improvement

Automation: Try to infer type of feature automatically with scripts and create feature engineering pipelines. This scales and can be used even when the features are not anonymized.


categorical variables v91 and v107 seems to be identical only different level names.
 
There are also some weird pattern between v22 and v125 ( isolate v22 and v125 and sort v22 reverse order). Any idea what to do with it ?
 
 
NN Methods
with my nn implementation best what I got is 5CV:0.468096 lb:0.46899. Its a 1000(relu)-1000(relu)-1(sigmoid) net with high dropout=0.9 in hidden layers, 110 epochs SGD. nn is very sensitive to data preprocessing, I think there is room for improvement. I use 1-hot for categories and rank trafo for numerics.

v72 is the exact sum of v129 v38 and v62