# RPSLearner: A novel approach combining Random Projection and Stacking Learning for categorizing NSCLC

Combining Random Projection and Stacking learning methods for lung cancer subtypes prediction based on RNA-seq data extracted from TCGA database.


## Usage
How to use the method for RNA-seq data

```python
# Usage Example for RPSLearner

import pandas as pd
from RPSLearner import RPSLearner
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('data/rnaseq_tcga.csv')

tpm = data.drop('Subtype', axis=1)
subtype = data['Subtype'] # Use '0' for LUAD, and '1' for LUSC

metrics = RPSLearner(
    tpm.values, subtype, n_jobs=5)
```
