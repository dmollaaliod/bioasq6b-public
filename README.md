# README #

### What is this repository for? ###

This code implements Macquarie University's experiments and
participation in BioASQ 6b.
* [Project page](http://web.science.mq.edu.au/~diego/medicalnlp/)
* [BioASQ](http://www.bioasq.org)

If you use this code, please cite the following paper:

D. Mollá. Macquarie University at BioASQ 6b: Deep learning and deep reinforcement learning for query-based multi-document summarisation (2018). Proceedings of the 6th BioASQ Workshop A challenge on large-scale biomedical semantic indexing and question answering. [[Proceedings](https://www.aclweb.org/anthology/W18-5303/)] [[arxiv](https://arxiv.org/abs/1809.05283)]

### How do I get set up? ###

Apart from the code in this repository, you will need the following files:

* `BioASQ-trainingDataset6b.json` - available from [BioASQ](http://www.bioasq.org/)
* `rouge_6b.csv` - you can create it by running the following overnight:
```
>>> from regression import saveRouge
>>> saveRouge('BioASQ-trainingDataset6b.json', 'rouge_6b.csv',
               snippets_only = True)
```
* `allMeSH_2016_100.vectors.txt` - ask diego.molla-aliod@mq.edu.au, or you can
use the continous space word vectors provided by [BioASQ](http://www.bioasq.org/) but
note that these are vectors with 200 dimensions, and the ones we need have
100 dimensions. The ones we generated used the same settings as the ones
provided by BioASQ, only with 100 dimensions.

Read the file `Dockerfile` for an idea of how to install the dependencies and
set up the system.

### Who do I talk to? ###

Diego Molla: [diego.molla-aliod@mq.edu.au](mailto:diego.molla-aliod@mq.edu.au)
