"""regressionlstm.py -- perform regression-based summarisation using LSTM

Author: Diego Molla <dmollaaliod@gmail.com>
Created: 24/8/2016
"""

import tensorflow as tf # required for running in NCI Raijin
import json
import codecs
import csv
import sys
import os
import shutil
import re
import random
import glob
from subprocess import Popen, PIPE
#from pathos.pools import ParallelPool as Pool
#from multiprocess import Pool
from multiprocessing import Pool
from functools import partial
import numpy as np

from sklearn.cross_validation import KFold

from nltk import sent_tokenize
from xml_abstract_retriever import getAbstract
from nnmodels import lstm, compare
import regressionbaselines
from summariser.basic import answersummaries

def bioasq_train(small_data=False, verbose=2):
    """Train model for BioASQ"""
    print("Training for BioASQ")
    regressor = Regression('BioASQ-trainingDataset6b.json',
                           'rouge_6b.csv',
                           nb_epoch=10,
                           verbose=verbose,
                           regression_type="LSTMSimilarities",
                           embeddings=True,
                           use_peepholes=False,
                           hidden_layer=50,
                           dropout=0.8,
                           batch_size=1024)
    indices = list(range(len(regressor.data)))
    if small_data:
        print("Training bioasq with small data")
        indices = indices[:20]
    regressor.train(indices, savepath="./task6b_nnr_model")

def bioasq_run(nanswers={"summary": 6,
                         "factoid": 2,
                         "yesno": 2,
                         "list": 3},
               test_data='phaseB_3b_01.json',
               output_filename='bioasq-out-nnr.json'):
    """Run model for BioASQ"""
    print("Running bioASQ")
    regressor = Regression('BioASQ-trainingDataset6b.json',
                           'rouge_6b.csv',
                           nb_epoch=10,
                           verbose=2,
                           regression_type="LSTMSimilarities",
                           embeddings=True,
                           use_peepholes=False,
                           hidden_layer=50,
                           dropout=0.8,
                           batch_size=1024)
    indices = list(range(len(regressor.data)))
    regressor.train(indices, savepath="./task6b_nnr_model", restore_model=True)
    testset = loaddata(test_data)
    answers = yield_bioasq_answers(regressor,
                                   testset,
                                   nanswers={"summary": 6,
                                             "factoid": 2,
                                             "yesno": 2,
                                             "list": 3})
    result = {"questions":[a for a in answers]}
    print("Saving results in file %s" % output_filename)
    with open(output_filename, 'w') as f:
        f.write(json.dumps(result, indent=2))
                                

    
def loaddata(filename):
    """Load the JSON data
    >>> data = loaddata('BioASQ-trainingDataset6b.json')
    Loading BioASQ-trainingDataset6b.json
    >>> len(data)
    2251
    >>> sorted(data[0].keys())
    ['body', 'concepts', 'documents', 'id', 'ideal_answer', 'snippets', 'type']
    """
    print("Loading", filename)
    data = json.load(open(filename, encoding="utf-8"))
    return data['questions']

def yield_candidate_text(questiondata, snippets_only=True):
    """Yield all candidate text for a question
    >>> data = loaddata("BioASQ-trainingDataset6b.json")
    Loading BioASQ-trainingDataset6b.json
    >>> y = yield_candidate_text(data[0], snippets_only=True)
    >>> next(y)
    ('55031181e9bde69634000014', 0, 'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes')
    >>> next(y)
    ('55031181e9bde69634000014', 1, "In this study, we review the identification of genes and loci involved in the non-syndromic common form and syndromic Mendelian forms of Hirschsprung's disease.")
    """
    past_pubmed = set()
    sn_i = 0
    for sn in questiondata['snippets']:
        if snippets_only:
            for s in sent_tokenize(sn['text']):
                yield (questiondata['id'], sn_i, s)
                sn_i += 1
            continue

        pubmed_id = os.path.basename(sn['document'])
        if pubmed_id in past_pubmed:
            continue
        past_pubmed.add(pubmed_id)
        file_name = os.path.join("Task6bPubMed", pubmed_id+".xml")
        sent_i = 0
        for s in sent_tokenize(getAbstract(file_name, version="0")[0]):
            yield (pubmed_id, sent_i, s)
            sent_i += 1

def yield_bioasq_answers(regressor, testset, nanswers=3):
    """Yield answer of each record for BioASQ shared task"""
    for r in testset:
        test_question = r['body']
        test_id = r['id']
        test_candidates = [(sent, sentid)
                           for (pubmedid, sentid, sent)
                           in yield_candidate_text(r)]
#        test_snippet_sentences = [s for snippet in r['snippets']
#                                  for s in sent_tokenize(snippet['text'])]
        if len(test_candidates) == 0:
            print("Warning: no text to summarise")
            test_summary = ""
        else:
            if isinstance(nanswers,dict):
                n = nanswers[r['type']]
            else:
                n = nanswers
            test_summary = " ".join(regressor.answersummaries([(test_question,
                                                                test_candidates,
                                                                n)])[0])
            #print("Test summary:", test_summary)
            
        if r['type'] == "yesno":
            exactanswer = "yes"
        else:
            exactanswer = ""

        yield {"id": test_id,
               "ideal_answer": test_summary,
               "exact_answer": exactanswer}

def collect_one_item(this_index, indices, testindices, data, rouge):
    "Collect one item for parallel processing"
    qi, d = this_index
    if qi in indices:
        partition = 'main'
    elif testindices != None and qi in testindices:
        partition = 'test'
    else:
        return None

    this_question = d['body']

    if 'snippets' not in d:
        return None
    data_snippet_sentences = [s for sn in d['snippets']
                              for s in sent_tokenize(sn['text'])]

    if len(data_snippet_sentences) == 0:
        return None

    candidates_questions = []
    candidates_sentences = []
    candidates_sentences_ids = []
    rouge_data = []
    for pubmed_id, sent_id, sent in yield_candidate_text(d):
        candidates_questions.append(this_question)
        candidates_sentences.append(sent)
        candidates_sentences_ids.append(sent_id)
        rouge_data.append(rouge[(qi, pubmed_id, sent_id)])

    return partition, rouge_data, candidates_questions, candidates_sentences, candidates_sentences_ids

class BaseRegression:
    """A base regression to be inherited"""
    def __init__(self, corpusFile, rougeFile, metric=['SU4']):
        """Initialise the regression system."""
        print("Reading data from %s and %s" % (corpusFile, rougeFile))
        self.data = loaddata(corpusFile)
        self.rouge = dict()
        with codecs.open(rougeFile, encoding='utf-8') as csv_file:
#        with codecs.open(rougeFile) as csv_file:
            reader = csv.DictReader(csv_file)
            lineno = 0
            num_errors = 0
            for line in reader:
                lineno += 1
                try:
                    key = (int(line['qid']), line['pubmedid'], int(line['sentid']))
                except:
                    num_errors += 1
                    print("Unexpected error:", sys.exc_info()[0])
                    print("%i %s" % (lineno, str(line).encode('utf-8')))
                else:
                    self.rouge[key] = np.mean([float(line[m]) for m in metric])
            if num_errors > 0:
                print("%i data items were ignored out of %i because of errors" % (num_errors, lineno))

    def extractfeatures(self, questions, candidates_sentences):
        """ Return the features"""
        assert len(questions) == len(candidates_sentences)

        # The following two lines are based on the text cleaning code of
        # BioASQ for generating word embeddings with word2vec
        cleantext = lambda t: re.sub(r'[.,?;*!%^&_+():-\[\]{}]', '',
                                     t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                   '').strip().lower()).split()

        return ([cleantext(sentence) for sentence in candidates_sentences],
                [cleantext(question) for question in questions])

    def _collect_data_(self, indices, testindices=None, resampling=None, resampling_test=None):
        """Collect the data given the question indices"""
        print("Collecting data")
        with Pool() as pool:
            collected = pool.map(partial(collect_one_item,
                                         indices=indices,
                                         testindices=testindices,
                                         data=self.data,
                                         rouge=self.rouge),
                                 enumerate(self.data))
        all_candidates_questions = {'main':[], 'test':[]}
        all_candidates_sentences = {'main':[], 'test':[]}
        all_candidates_sentences_ids = {'main':[], 'test':[]}
        all_rouge = {'main':[], 'test':[]}
        for c in collected:
            if c == None:
                continue
            partition, rouge_data, candidates_questions, candidates_sentences, candidates_sentences_ids = c
            all_candidates_questions[partition] += candidates_questions
            all_candidates_sentences[partition] += candidates_sentences
            all_candidates_sentences_ids[partition] += candidates_sentences_ids
            all_rouge[partition] += rouge_data

        if resampling is not None:
            # resample from the training features to obtain balanced data
            bins = [[], [], [], [], [], [], [], [], [], []]
            for i, su4 in enumerate(all_rouge['main']):
                assert 0.0 <= su4 and su4 <= 1.0
                bin_i = int(su4 * 10)
                if su4 == 1.0:
                    bin_i = 9
                bins[bin_i].append(i)
            print("Histogram of ROUGE before sampling %i samples per training bin:" % resampling)
            sampled_i = []
            for i in range(10):
                print(len(bins[i]))
                if len(bins[i]) > 0:
                    samples = np.random.choice(len(bins[i]), resampling)
                    sampled_i += [bins[i][j] for j in samples]
            random.shuffle(sampled_i)
            all_candidates_questions['main'] = [all_candidates_questions['main'][i] for i in sampled_i]
            all_candidates_sentences['main'] = [all_candidates_sentences['main'][i] for i in sampled_i]
            all_candidates_sentences_ids['main'] = [all_candidates_sentences_ids['main'][i] for i in sampled_i]
            all_rouge['main'] = [all_rouge['main'][i] for i in sampled_i]

        if resampling_test is not None:
            # resample from the test features to obtain balanced data
            bins = [[], [], [], [], [], [], [], [], [], []]
            for i, su4 in enumerate(all_rouge['test']):
                assert 0.0 <= su4 and su4 <= 1.0
                bin_i = int(su4 * 10)
                if su4 == 1.0:
                    bin_i = 9
                bins[bin_i].append(i)
            print("Histogram of ROUGE before sampling %i samples per test bin:" % resampling_test)
            sampled_i = []
            for i in range(10):
                print(len(bins[i]))
                if len(bins[i]) > 0:
                    samples = np.random.choice(len(bins[i]), resampling_test)
                    sampled_i += [bins[i][j] for j in samples]
            random.shuffle(sampled_i)
            all_candidates_questions['test'] = [all_candidates_questions['test'][i] for i in sampled_i]
            all_candidates_sentences['test'] = [all_candidates_sentences['test'][i] for i in sampled_i]
            all_candidates_sentences_ids['test'] = [all_candidates_sentences_ids['test'][i] for i in sampled_i]
            all_rouge['test'] = [all_rouge['test'][i] for i in sampled_i]

        print("End collecting data")
        return all_rouge, all_candidates_questions, all_candidates_sentences, all_candidates_sentences_ids
        
class Regression(BaseRegression):
    """A regression system"""
    def __init__(self, corpusFile, rougeFile, metric=['SU4'],
                 nb_epoch=3, verbose=2,
                 regression_type="Bi-LSTM",
                 embeddings=True,
                 use_peepholes=False,
                 hidden_layer=0,
                 dropout=0.5,
                 batch_size=128):
        """Initialise the regression system."""
        BaseRegression.__init__(self, corpusFile, rougeFile, metric)
        self.nb_epoch = nb_epoch
        self.use_peepholes = use_peepholes
        self.verbose = verbose
        self.dropout = dropout

        self.lstm = None
        if regression_type == "BasicNN":
            self.lstm = lstm.BasicNN(embeddings=embeddings,
                                    hidden_layer=hidden_layer,
                                    batch_size=batch_size)
        elif regression_type == "CNN":
            self.lstm = lstm.CNN(embeddings=embeddings,
                                 hidden_layer=hidden_layer,
                                 batch_size=batch_size)
        elif regression_type == "LSTM":
            self.lstm = lstm.LSTM(embeddings=embeddings,
                                  hidden_layer=hidden_layer,
                                  batch_size=batch_size)
        elif regression_type == "Bi-LSTM":
            self.lstm = lstm.LSTMBidirectional(embeddings=embeddings,
                                               hidden_layer=hidden_layer,
                                               batch_size=batch_size)
        elif regression_type == "Similarities":
            self.lstm = lstm.Similarities(embeddings=embeddings,
                                          hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimMul(),
                                          positions=True)
        elif regression_type == "SimilaritiesYu":
            self.lstm = lstm.Similarities(embeddings=embeddings,
                                          hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimYu(),
                                          positions=True)
        elif regression_type == "SimilaritiesEuc":
            self.lstm = lstm.Similarities(embeddings=embeddings,
                                          hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimEuc(),
                                          positions=True)
        elif regression_type == "CNNSimilarities":
            self.lstm = lstm.CNNSimilarities(embeddings=embeddings,
                                             hidden_layer=hidden_layer,
                                             batch_size=batch_size,
                                             comparison = compare.SimMul())
        elif regression_type == "CNNSimilaritiesYu":
            self.lstm = lstm.CNNSimilarities(embeddings=embeddings,
                                             hidden_layer=hidden_layer,
                                             batch_size=batch_size,
                                             comparison = compare.SimYu())
        elif regression_type == "CNNSimilaritiesEuc":
            self.lstm = lstm.CNNSimilarities(embeddings=embeddings,
                                             hidden_layer=hidden_layer,
                                             batch_size=batch_size,
                                             comparison = compare.SimEuc())
        elif regression_type == "LSTMSimilarities":
            self.lstm = lstm.LSTMSimilarities(embeddings=embeddings,
                                              hidden_layer=hidden_layer,
                                              batch_size=batch_size,
                                              comparison = compare.SimMul(),
                                              positions=True)
        elif regression_type == "LSTMSimilaritiesYu":
            self.lstm = lstm.LSTMSimilarities(embeddings=embeddings,
                                              hidden_layer=hidden_layer,
                                              batch_size=batch_size,
                                              comparison = compare.SimMul(),
                                              positions=True)
        elif regression_type == "LSTMSimilaritiesEuc":
            self.lstm = lstm.LSTMSimilarities(embeddings=embeddings,
                                              hidden_layer=hidden_layer,
                                              batch_size=batch_size,
                                              comparison = compare.SimEuc(),
                                              positions=True)
        elif args.regression_type == "TfidfNNR":
            self.lstm = regressionbaselines.TfidfNNR(batch_size=args.batch_size,
                                                 n_components=args.svd_components)
        elif args.regression_type == "TfidfSimNNR":
            self.lstm = regressionbaselines.TfidfSimNNR(batch_size=args.batch_size,
                                                 n_components=args.svd_components)
        elif args.regression_type == "TfidfSim2NNR":
            self.lstm = regressionbaselines.TfidfSim2NNR(batch_size=args.batch_size,
                                                 n_components=args.svd_components,
                                                 comparison=compare.SimMul())
        elif args.regression_type == "TfidfSimYuNNR":
            self.lstm = regressionbaselines.TfidfSim2NNR(batch_size=args.batch_size,
                                                 n_components=args.svd_components,
                                                 comparison=compare.SimYu())
        elif args.regression_type == "TfidfSimEucNNR":
            self.lstm = regressionbaselines.TfidfSim2NNR(batch_size=args.batch_size,
                                                 n_components=args.svd_components,
                                                 comparison=compare.SimEuc())

    def train(self, indices, testindices=None, foldnumber=0, restore_model=False,
              savepath=None,
              save_test_predictions=None, resampling=None, resampling_test=None):
        """Train the regressor given the question indices"""
        print("Gathering training data")
        if savepath is None:
            savepath="savedmodels/%s_%i" % (self.lstm.name(), foldnumber)
        all_rouge, candidates_questions, candidates_sentences, candidates_sentences_ids = \
        self._collect_data_(indices, testindices, resampling=resampling, resampling_test=resampling_test)

        features = self.extractfeatures(candidates_questions['main'],
                                        candidates_sentences['main'])
        
        print("Training %s" % self.lstm.name())
        loss_train = self.lstm.fit(features[0], features[1],
                                   np.array([[r] for r in all_rouge['main']]),
                                   X_positions=np.array([[cid] for cid in candidates_sentences_ids['main']]),
                                   nb_epoch=self.nb_epoch,
                                   use_peepholes=self.use_peepholes,
                                   verbose=self.verbose,
                                   dropoutrate=self.dropout,
                                   savepath=savepath,
                                   restore_model=restore_model)
        if testindices == None:
            return loss_train
        else:
            features_test = self.extractfeatures(candidates_questions['test'],
                                                 candidates_sentences['test'])
            if save_test_predictions is None:
                loss_test = self.lstm.test(features_test[0],
                                           features_test[1],
                                           np.array([[r] for r in all_rouge['test']]),
                                           X_positions=np.array([[cid] for cid in candidates_sentences_ids['test']]))
            else:
                predictions_test = self.lstm.predict(features_test[0],
                                                     features_test[1],
                                                     X_positions=[[cid] for cid in candidates_sentences_ids['test']])
                predictions_test = [p[0] for p in predictions_test]
                print("Saving predictions in %s" % save_test_predictions)
                with open(save_test_predictions, "w") as f:
                    writer = csv.DictWriter(f, fieldnames=["id", "target", "prediction"])
                    writer.writeheader()
                    for i, p in enumerate(predictions_test):
                        writer.writerow({"id": i,
                                         "target": all_rouge['test'][i],
                                         "prediction": p})
                print("Predictions saved")
                loss_test = np.mean((predictions_test - np.array(all_rouge['test'])) ** 2)
            return loss_train, loss_test

    def test(self, indices):
        """Test the regressor given the question indices"""
        print("Gathering test data")
        all_rouge, candidates_questions, candidates_sentences, candidates_sentences_ids = \
        self._collect_data_(indices)

        features = self.extractfeatures(candidates_questions['main'],
                                        candidates_sentences['main'])

        print("Testing LSTM")
        loss = self.lstm.test(features[0], 
                              features[1],
                              [[r] for r in all_rouge['main']],
                              X_positions=[[cid] for cid in candidates_sentences_ids['main']])
        print("MSE = %f" % loss)
        return loss

    def answersummaries(self, questions_and_candidates, beamwidth=0):
        if beamwidth > 0:
            print("Beam width is", beamwidth)
            return answersummaries(questions_and_candidates, self.extractfeatures, self.lstm.predict, beamwidth)
        else:
            return answersummaries(questions_and_candidates, self.extractfeatures, self.lstm.predict)

    def answersummary(self, question, candidates_sentences,
                      n=3, qindex=None):
        """Return a summary that answers the question

        qindex is not used but needed for compatibility with oracle"""
        return self.answersummaries((question, candidates_sentences, n))
        

def evaluate_one(di, dataset, testindices, nanswers, rougepath):
    """Evaluate one question"""
    if di not in testindices:
        return None
    question = dataset[di]['body']
    if 'snippets' not in dataset[di].keys():
        return None
    candidates = [(sent, sentid) for (pubmedid, sentid, sent) in yield_candidate_text(dataset[di])]
    if len(candidates) == 0:
        # print("Warning: No text to summarise; ignoring this text")
        return None

    if type(nanswers) == dict:
        n = nanswers[dataset[di]['type']]
    else:
        n = nanswers
    rouge_text = """<EVAL ID="%i">
 <MODEL-ROOT>
 %s/models
 </MODEL-ROOT>
 <PEER-ROOT>
 %s/summaries
 </PEER-ROOT>
 <INPUT-FORMAT TYPE="SEE">
 </INPUT-FORMAT>
""" % (di, rougepath, rougepath)
    rouge_text += """ <PEERS>
  <P ID="A">summary%i.txt</P>
 </PEERS>
 <MODELS>
""" % (di)

    if type(dataset[di]['ideal_answer']) == list:
        ideal_answers = dataset[di]['ideal_answer']
    else:
        ideal_answers = [dataset[di]['ideal_answer']]

    for j in range(len(ideal_answers)):
        rouge_text += '  <M ID="%i">ideal_answer%i_%i.txt</M>\n' % (j,di,j)
        with codecs.open(rougepath + '/models/ideal_answer%i_%i.txt' % (di,j),
                         'w', 'utf-8') as fout:
            a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</body>\n</html>'.format(ideal_answers[j])
            fout.write(a+'\n')
    rouge_text += """ </MODELS>
</EVAL>
"""
    target = {'id': dataset[di]['id'],
              'ideal_answer': ideal_answers,
              'exact_answer': ""}
    return rouge_text, di, (question, candidates, n), target
    

def evaluate(regressionClassInstance, rougeFilename="rouge.xml", nanswers=3,
             tmppath='', load_models=False, small_data=False, fold=0):
    """Evaluate a regression-based summariser

    nanswers is the number of answers. If it is a dictionary, then the keys indicate the question type, e.g.
    nanswers = {"summary": 6,
                "factoid": 2,
                "yesno": 2,
                "list": 3}
"""
    if tmppath == '':
        modelspath = 'saved_models_Similarities'
        rougepath = '../rouge'
        crossvalidationpath = 'crossvalidation'
    else:
        modelspath = tmppath + '/saved_models'
        rougepath = tmppath + '/rouge'
        crossvalidationpath = tmppath + '/crossvalidation'
        rougeFilename = rougepath + "/" + rougeFilename
        if not os.path.exists(rougepath):
            os.mkdir(rougepath)
        for f in glob.glob(rougepath + '/*'):
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
            else:
                print("Warning: %f is neither a file nor a directory" % (f))
        os.mkdir(rougepath + '/models')
        os.mkdir(rougepath + '/summaries')
        if not os.path.exists(crossvalidationpath):
            os.mkdir(crossvalidationpath)

    dataset = regressionClassInstance.data
    indices = [i for i in range(len(dataset))
               #if dataset[i]['type'] == 'summary'
               #if dataset[i]['type'] == 'factoid'
               #if dataset[i]['type'] == 'yesno'
               #if dataset[i]['type'] == 'list'
               ]
    if small_data:
        indices = indices[:100]
        # resampling = 10
        # resampling_test = 5
        resampling = None
        resampling_test = None
    else:
#        resampling = 1000
#        resampling_test = 100
        resampling = None
        resampling_test = None

    random.seed(1234)
    random.shuffle(indices)

    rouge_results = []
    the_fold = 0
    for (traini, testi) in KFold(len(indices), n_folds=10):
        the_fold += 1

        if fold > 0 and the_fold != fold:
            continue

        if small_data and the_fold > 2:
           break

        print("Cross-validation Fold %i" % the_fold)
        trainindices = [indices[i] for i in traini]
        testindices = [indices[i] for i in testi]

        save_test_predictions = crossvalidationpath + "/test_results_%i.csv" % the_fold
        (trainloss,testloss) = regressionClassInstance.train(trainindices,
                                                             testindices,
                                                             foldnumber=the_fold,
                                                             restore_model=load_models,
                                                             savepath="%s/saved_model_%i" % (modelspath, the_fold),
                                                             save_test_predictions=save_test_predictions,
                                                             resampling = resampling,
                                                             resampling_test=resampling_test)

        for f in glob.glob(rougepath+'/models/*')+glob.glob(rougepath+'/summaries/*'):
            os.remove(f)

        with open(rougeFilename,'w') as frouge:
           print("Collecting evaluation results")
           frouge.write('<ROUGE-EVAL version="1.0">\n')
           with Pool() as pool:
               evaluation_data = \
                  pool.map(partial(evaluate_one,
                                   dataset=dataset,
                                   testindices=testindices,
                                   nanswers=nanswers,
                                   rougepath=rougepath),
                           range(len(dataset)))
                           
           summaries = regressionClassInstance.answersummaries([e[2] for e in evaluation_data if e != None])
                           
           eval_test_system = []
           eval_test_target = []
           for data_item in evaluation_data:
               if data_item == None:
                   continue
               rouge_item, di, system_item, target_item = data_item
               summary = summaries.pop(0)
               #print(di)
               with codecs.open(rougepath+'/summaries/summary%i.txt' % (di),
                               'w', 'utf-8') as fout:
                   a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</body>\n</html>'.format(" ".join(summary))
                   fout.write(a+'\n')
                   # fout.write('\n'.join([s for s in summary])+'\n')

               frouge.write(rouge_item)
               system_item = {'id': dataset[di]['id'],
                              'ideal_answer': " ".join(summary),
                              'exact_answer': ""}

               eval_test_system.append(system_item)
               eval_test_target.append(target_item)
               
           assert len(summaries) == 0
           
           frouge.write('</ROUGE-EVAL>\n')

        json_summaries_file = crossvalidationpath + "/crossvalidation_%i_summaries.json" % the_fold
        print("Saving summaries in file %s" % json_summaries_file)
        with open(json_summaries_file,'w') as fcv:
            fcv.write(json.dumps({'questions': eval_test_system}, indent=2))
        json_gold_file = crossvalidationpath + "/crossvalidation_%i_gold.json" % the_fold
        print("Saving gold data in file %s" % json_gold_file)
        with open(json_gold_file,'w') as fcv:
            fcv.write(json.dumps({'questions': eval_test_target}, indent=2))

        print("Calling ROUGE", rougeFilename)
        ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ' \
            + '../rouge/RELEASE-1.5.5/data -c 95 -2 4 -u -x -n 4 -a ' \
            + rougeFilename
        stream = Popen(ROUGE_CMD, shell=True, stdout=PIPE).stdout
        lines = stream.readlines()
        stream.close()
        for l in lines:
            print(l.decode('ascii').strip())
        print()

        F = {'N-1':float(lines[3].split()[3]),
             'N-2':float(lines[7].split()[3]),
             'L':float(lines[11].split()[3]),
             'S4':float(lines[15].split()[3]),
             'SU4':float(lines[19].split()[3]),
             'trainloss':trainloss,
             'testloss':testloss}
        rouge_results.append(F)

        print("N-2: %1.5f SU4: %1.5f TrainMSE: %1.5f TestMSE: %1.5f" % (
               F['N-2'], F['SU4'], F['trainloss'], F['testloss']
        ))


    print("%5s %7s %7s %7s %7s" % ('', 'N-2', 'SU4', 'TrainMSE', 'TestMSE'))
    for i in range(len(rouge_results)):
        print("%5i %1.5f %1.5f %1.5f %1.5f" % (i+1,rouge_results[i]['N-2'],rouge_results[i]['SU4'],
                                       rouge_results[i]['trainloss'],rouge_results[i]['testloss']))
    mean_N2 = np.average([rouge_results[i]['N-2']
                          for i in range(len(rouge_results))])
    mean_SU4 = np.average([rouge_results[i]['SU4']
                           for i in range(len(rouge_results))])
    mean_Trainloss = np.average([rouge_results[i]['trainloss']
                                 for i in range(len(rouge_results))])
    mean_Testloss = np.average([rouge_results[i]['testloss']
                                for i in range(len(rouge_results))])
    print("%5s %1.5f %1.5f %1.5f %1.5f" % ("mean",mean_N2,mean_SU4,mean_Trainloss,mean_Testloss))
    stdev_N2 = np.std([rouge_results[i]['N-2']
                       for i in range(len(rouge_results))])
    stdev_SU4 = np.std([rouge_results[i]['SU4']
                        for i in range(len(rouge_results))])
    stdev_Trainloss = np.std([rouge_results[i]['trainloss']
                              for i in range(len(rouge_results))])
    stdev_Testloss = np.std([rouge_results[i]['testloss']
                             for i in range(len(rouge_results))])
    print("%5s %1.5f %1.5f %1.5f %1.5f" % ("stdev",stdev_N2,stdev_SU4,stdev_Trainloss,stdev_Testloss))
    print()
    return mean_SU4, stdev_SU4, mean_Testloss, stdev_Testloss

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    #import sys
    # #bioasq_train()
    # bioasq_run()
    #sys.exit()
    
    import argparse
    import time
    import socket
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--nb_epoch', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('-v', '--verbose', type=int, default=2,
                        help="Verbosity level")
    parser.add_argument('-t', '--regression_type',
                        choices=("BasicNN", "CNN", "LSTM", "Bi-LSTM",
                                 "Similarities", "CNNSimilarities", "LSTMSimilarities",
                                 "SimilaritiesYu", "CNNSimilaritiesYu", "LSTMSimilaritiesYu",
                                 "SimilaritiesEuc", "CNNSimilaritiesEuc", "LSTMSimilaritiesEuc",
                                 "TfidfNNR", "TfidfSimNNR", "TfidfSim2NNR", 
                                 "TfidfSimYuNNR", "TfidfSimEucNNR"),
                        default="Similarities",
                        help="Type of regression")
    parser.add_argument('-S', '--small', action="store_true",
                        help='Run on a small subset of the data')
    parser.add_argument('-l', '--load', action="store_true",
                        help='load pre-trained model')
    parser.add_argument('-m', '--embeddings', action="store_true",
                        help="Use pre-trained embeddings")
    parser.add_argument('-p', '--peepholes', action="store_true",
                        help="Use peepholes in the LSTM models")
    parser.add_argument('-d', '--hidden_layer', type=int, default=50,
                        help="Size of the hidden layer (0 if there is no hidden layer)")
    parser.add_argument('-r', '--dropout', type=float, default=0.9,
                        help="Keep probability for the dropout layers")
    parser.add_argument('-a', '--tmppath', default='',
                        help="Path for temporary data and files")
    parser.add_argument('-s', '--batch_size', type=int, default=4096,
                        help="Batch size for gradient descent")
    parser.add_argument('-c', '--svd_components', type=int, default=100,
                        help="Depth of the LSTM stack")
    parser.add_argument("-f", "--fold", type=int, default=0,
                        help="Use only the specified fold (0 for all folds)")

    args = parser.parse_args()

    if args.tmppath != '':
        lstm.DB = "%s/%s" % (args.tmppath, lstm.DB)


    regressor = Regression('BioASQ-trainingDataset6b.json',
                           'rouge_6b.csv',
                           nb_epoch=args.nb_epoch,
                           verbose=args.verbose,
                           regression_type=args.regression_type,
                           embeddings=args.embeddings,
                           use_peepholes=args.peepholes,
                           hidden_layer=args.hidden_layer,
                           dropout=args.dropout,
                           batch_size=args.batch_size)

    print("%s with epochs=%i and batch size=%i" % (regressor.lstm.name(), 
                                                   args.nb_epoch,
                                                   args.batch_size))


    mean_SU4, stdev_SU4, mean_Testloss, stdev_Testloss = \
                              evaluate(regressor,
                                       nanswers={"summary": 6,
                                                 "factoid": 2,
                                                 "yesno": 2,
                                                 "list": 3},
                                       tmppath=args.tmppath,
                                       load_models=args.load,
                                       small_data=args.small,
                                       fold = args.fold)
    end_time = time.time()
    elapsed = time.strftime("%X", time.gmtime(end_time - start_time))
    print("Time elapsed: %s" % (elapsed))
    print("| Type | Epochs | Dropout | meanSU4 | stdevSU4 | meanTestLoss | stdevTestLoss | Time | Hostname |")
    if args.peepholes:
        str_peepholes = "peepholes"
    else:
        str_peepholes = ""
    print("| %s %s | %i | %f | %f | %f | %f | %f | %s | %s |" % \
               (regressor.lstm.name(),
                str_peepholes,
                args.nb_epoch,
                args.dropout,
                mean_SU4,
                stdev_SU4,
                mean_Testloss,
                stdev_Testloss,
                elapsed,
                socket.gethostname()))
