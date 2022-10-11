# Conditional Graph Information Bottleneck for Relational Learning
The official source code for Conditional Graph Information Bottleneck for Relational Learning (CGIB).

**We uploaded online appendix which contains the information about used datasets in the experiments [**(Link)**](https://anonymous.4open.science/r/CGIB-AB1F/CGIB_online_appendix.pdf).**

## Overview
Relational learning, whose goal is to learn the interaction behavior between entity pairs, plays a central role in various machine learning tasks such as similarity learning and interaction prediction. In molecular science, graph neural networks have recently shown great success in relational learning by modeling a molecule as a graph structure, and considering atom-level interactions between two molecules. Despite their success, existing relational learning methods tend to overlook the nature of chemicals, i.e., a chemical is composed of multiple substructures such as functional groups that cause distinctive chemical reactions. In this work, we propose a novel relational learning method, called CGIB, that predicts the interaction behavior between a pair of graphs by detecting important subgraphs therein. The main idea is, given a pair of graphs, to find a substructure from a graph that contains the minimal sufficient information regarding the task at hand conditioned on the paired graph based on the principle of conditional graph information bottleneck. We argue that our proposed method mimics the nature of chemical reactions, i.e., the core substructure of a molecule varies depending on which other molecule it interacts with. Extensive experiments on various tasks with real-world datasets demonstrate the superiority of CGIB over state-of-the-art baselines.

<img src="imgs/Figure1.png" width="500px"></img> 

**a) Molecules with hydroxyl group have high aqueous solubility. (b) C-CF3 structure decreases the solubility of molecules in water. (c) However, C-CF3 structure may not be crucial in determining the solubility of molecules in oil.**

### Requirements

- Python version: 3.7.10
- Pytorch version: 1.8.1
- torch-geometric version: 1.7.0