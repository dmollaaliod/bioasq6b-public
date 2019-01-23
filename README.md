# README #

This README would normally documents the steps necessary to install the code.

### What is this repository for? ###

This code implements Macquarie University's experiments and
participation in BioASQ 6b.
* [BioASQ](http://www.bioasq.org)
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

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
