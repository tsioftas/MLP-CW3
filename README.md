In this project, we developed a CNN model to predic weather temperature on heterogenous tabular data. In addition to that, the model was compared with other deep learning models such as fully connected layer model and stripped-down CNN and also with CatBoost decision tree.

Dataset: We used the data set provided by Yandex. It features the weather data during the Yandex shifts challenge, which took place in October-November
2021. Our data (Dat) can be characterized as heterogeneous and tabular. Heterogeneous means that it comes from various sources (e.g. measurements of different physical quantities, output from weather prediction models and more). Heterogenous data contrasts with homogeneous data, which comes from a single source. Tabular data means it can be expressed as a table. One fundamental property of tabular data is that there is little to no correlation between consecutive rows, unlike, for example, time-series data or indexed data. The task at hand is to build a model that accurately finds the
temperature at a specific location at a specific time, given a set of features characterizing that location at that time (e.g. pressure, humidity etc.). The model needs to work well regardless of time and location. The dataset was partitioned into training, development and evaluation sets. There are about 3 million data points in the training partition, 100 000 in development and about 1 million in the evaluation partition. The development and evaluation partitions were further split in half into in-domain, and out-of-domain data, that is, data from climates and regions that appear in the training set and data from climates and regions that do not.


 Pre-Processing: Some pre-processing is applied to feature data:

• Imputing: The missing values are replaced in all in-
put columns following a simple constant strategy (fill
value is -1).

• Quantisation: Each input column is discretised into
100 quantile bins; then, the bin identifier can be con-
sidered a quantised numerical value of the original
feature; The result is a normal distribution.

• Standardisation: Each quantised column is standard-
ised by removing the mean and scaling to unit vari-
ance.

• PCA: Principal Component Analysis is applied and
95% of the variance is retained. These are concate-
nated to the original data to help with grouping fea-
tures together.

Model Description: Our model is a CNN that creates images from non-image data. The idea with tabular data is that we could create space for features in a fully connected layer since it is hard to detect spatial correlation. The fully connected layer is learning possible nonlinear functions in that space. Adding such a layer is a (usually) inexpensive way to learn nonlinear combinations of advanced features. After the first dense layer, we regroup the features in 16x1 feature maps, which we feed into the next layer, which is convolutional.We also applied RELU and pooling layers to the output of the convolution layers. 

Evaluation Metrics: To evaluate model's performance, we udr the mean squared error (MSE), root mean squared error (RMSE) and mean absolute error(MAE) of all the data points in the evaluation dataset. The lower the value, the better, which implies that the predicted value is closer to the actual value.

