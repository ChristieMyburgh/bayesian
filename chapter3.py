import numpy as np
import pymc3 as pm
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.tail())

sns.violinplot(x='day', y='tip', data=tips)

y = tips['tip'].values
x = pd.Categorical(tips['day']).codes

with pm.Model() as comparing_groups:
	means = pm.Normal('means', mu=0, sd=10, shape=len(set(x)))
	sds = pm.HalfNormal('sds', sd=10, shape=len(set(x)))

	y = pm.Normal('y', mu=means[x], sd=sds[x], observed=y)

	trace_cg = pm.sample(5000)

chain_cg = trace_cg[100::]
pm.traceplot(chain_cg)
plt.show()

