# <span style="color:#F7B617">Py</span><span style="color:#c82804">CIPHOD</span> - <span style="color:#4851a1">Python package for Causal Inference in Public Health using Observational Databases</span> (Beta)


## ⚠️ Status
- Beta version (v0.1)
- This package is under active development: not all functionalities are available
- Disclaimer: This package is under development and may change. Use with caution applications.



## Introduction

PyCIPHOD is a robust and versatile Python package developed and maintained by the CIPHOD team at the Pierre Louis Institute of Epidemiology and Public Health, INSERM, Sorbonne University. It is designed to empower epidemiologists and practitioners to uncover and understand intricate causal mechanisms in their data. PyCIPHOD offers a comprehensive suite of methods for causal inference, including causal discovery, causal reasoning, and root cause analysis, and is suited for both fully specified and partially specified graphs. It effectively handles temporal data, such as time series or cohort data, addressing the evolving needs of epidemiologists.




## Key features

### Data type
* iid data
* Time series
* Cohorts

### Temporal graphs
* Fully specified graphs
* Partially specified graphs: partially oriented graphs, cluster graphs, summary causal graphs, difference graphs


### Causal discovery
* Classical causal discovery:  Learn the entire causal structure of the dataset, which can be computationally intensive and may not always be necessary for specific research questions.

* Local causal discovery: Focus on discovering causal relations in a specific part of the causal graph, particularly around a target variable (such as a treatment or outcome), rather than attempting to learn the entire causal structure of the dataset. This approach is computationally more efficient because it limits the scope to the local neighborhood of the target variable.

* Federated causal discovery: Harness the power of distributed data sources while preserving data privacy.

* Difference graph discovery: Identify changes in causal relationships over time by comparing causal graphs at different time points, enabling the detection of temporal shifts in causality.


### Causal reasoning and estimation
* Determine whether a causal effect can be uniquely computed from the available data and the structure of the causal graph.
* If the causal effect can be uniquely derived, obtain an estimand to be used for estimating the causal effect from the data. 
* Apply statistical methods to estimate the estimand from the available data.
* Search optimal 

### Root cause analysis
* Identify and analyze the root causes of observed anomalies by using causal graphs to trace the origins of effects within your data. 
* Receive recommendations on where to intervene to eliminate the anomalies.


## Quick Start


## More Information & Resources
About causal discovery
* https://openreview.net/pdf?id=PGLbZpVk2n
* https://openreview.net/forum?id=TcMKrmLJCL

About causal reasoning
* https://openreview.net/forum?id=lL4EzE8bY8
* https://ojs.aaai.org/index.php/AAAI/article/view/30021
* https://openreview.net/forum?id=5f7YlSKG1l
* https://ojs.aaai.org/index.php/AAAI/article/view/34882
* https://openreview.net/forum?id=yYGdFo4kKb
* https://openreview.net/forum?id=905LEugq6R

About root cause analysis
* https://proceedings.mlr.press/v206/assaad23a/assaad23a.pdf
* https://dl.acm.org/doi/pdf/10.1145/3627673.3680010


## List of all contributors (alphabetical order)
* Charles Assaad
* Federico Baldo
* Simon Ferreira
* Timothée Loranchet
