import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from KNNImplementation import *
from DecisionTreeClassifier import *

data = pd.read_csv('Heart_Disease_Dataset.csv')
data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']

print(data.isnull().sum())
print(data.describe())

data['sex'] = data['sex'].map({0: 'female', 1: 'male'})

# Graph based on age
sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
sns.catplot(kind = 'count', data = data, x = 'age', hue = 'target', order = data['age'].sort_values().unique())
plt.title('Age Vs target count')
# mpl.use("Qt5Agg")
# plt.get_current_fig_manager().window.showMaximized()
plt.show()

# Graph based on gender
sns.catplot(kind = 'bar', data = data, y = 'Target Count', x = 'sex', hue = 'target')
plt.title('Gender Vs target count')
plt.show()

# Data Preprocessing
data['sex'] = data.sex.map({'female': 0, 'male': 1})

kNNClassifier(data)

decisionTreeClassifier(data)


