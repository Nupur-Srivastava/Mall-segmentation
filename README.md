# K means-Machine learning
## Model Evaluation and Validation
## Project:Mall-segmentation

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

### Code
Template code is provided in the `mall-segmentation.ipynb` notebook file. You will also be required to use the included Python file and dataset file to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project. If you are interested in how the visualizations are created in the notebook, please feel free to explore this Python file.
### Run

In a terminal or command window, navigate to the top-level project directory (that's there in the project) and run one of the following commands:

```bash
ipython notebook digits recongiation.ipynb
```  
or
```bash
jupyter notebook digits recongiation.ipynb
```
or open with Juoyter Lab
```bash
jupyter lab
```
This will open the Jupyter Notebook software and project file in your browser.

## K-means
K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. ... Data points are clustered based on feature similarity
## Elbow method
The elbow method runs k-means clustering on the dataset for a range of values for k (say from 1-10) and then for each value of k computes an average score for all clusters
![download (1)](https://user-images.githubusercontent.com/70944857/136671994-c6740896-76b3-49e0-9c2e-75e7f36714d3.png)

## Data:
The data set is taken from [Kaggle](https://www.kaggle.com/).Its a platform that provide datsets for free and help to do experiment.

**Features**
1.  `Gender`: gender of the person
2. `Annual income`: annual income of the person
3. `Spending Score`: the spending score of the person
4. `Age`:Age of the person

**Target Variable**

5. `Spending score`: the spending score of the person
6. `Annual income`:annual income of the person
