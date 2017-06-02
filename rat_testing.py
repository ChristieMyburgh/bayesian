import numpy as np
import pymc3 as pm
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

tumers = [
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,
    1,1,2,2,2,2,2,2,2,2,
    2,1,5,2,5,3,2,7,7,3,
    3,2,9,10,4,4,4,4,4,4,
    4,10,4,4,4,5,11,12,5,5,
    6,5,6,6,6,6,16,15,15,9
]

n = [
    20,20,20,20,20,20,20,19,19,19,
    19,18,18,17,20,20,20,20,19,20,
    18,18,25,24,23,20,20,20,20,20,
    20,10,49,19,46,27,17,49,47,20,
    20,13,48,50,20,20,20,20,20,20,
    20,48,19,19,19,22,46,49,20,20,
    23,19,22,20,20,20,52,47,46,24
    ]

group_idx = np.repeat(np.arange(len(n)), n)

data=[]

for i in range(0, len(n)):
    data.extend(np.repeat([1, 0], [tumers[i], n[i] - tumers[i]]))

pm_rat = pm.Model()

with pm_rat:
	# Hyper priors
	alpha = pm.HalfCauchy('alpha', beta=10)

	beta = pm.HalfCauchy('beta', beta=10)

	# Prior distribution - beta with alpha, beta params
	theta = pm.Beta('theta', alpha, beta, shape=len(n))

	# Likelihood - Independent binomial distribution
	y = pm.Bernoulli('y', p=theta[group_idx], observed=data)

	trace = pm.sample(2000)

chain_r = trace[200:]
pm.traceplot(chain_r)

x = np.linspace(0, 1, 100)

for i in np.random.randint(0, len(chain_r), size=100):
	pdf = stats.beta(chain_r['alpha'][i], chain_r['beta'][i]).pdf(x)
	plt.plot(x, pdf, 'g', alpha=0.05)

dist = stats.beta(chain_r['alpha'].mean(), chain_r['beta'].mean())
pdf = dist.pdf(x)
mode = x[np.argmax(pdf)]
mean = dist.moment(1)
plt.plot(x, pdf, label='mode = {:.2f}\nmean = {:.2f}'.format(mode, mean))

plt.legend(fontsize=14)
plt.xlabel(r'$\theta_{prior}$', fontsize=16)

y_pred = pm.sample_ppc(chain_r, 100, pm_rat, size=len(data))

print("Done...")