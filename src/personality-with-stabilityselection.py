# coding: utf-8
import argparse
import codecs
import os
import sys
import random
import pandas as pd
import cPickle
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from features import Featurizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
#from sklearn.naive_bayes import BernoulliNB

parser = argparse.ArgumentParser(description="Authorship attribution")
parser.add_argument('data', help="data")
parser.add_argument('--coef', help="print coeficients", action='store_true', default=False)
parser.add_argument('--meta', help="file with meta data", default=None, type=str)
parser.add_argument('--dim', help="label dimension (default: all)", default=None, type=int)
parser.add_argument('--folds', help="number of folds for CV (default 10)", default=10, type=int)
parser.add_argument('--seed', help="random seed",type=int,default=1234)
parser.add_argument('--txt', help="tab separated txt file (default: json)", action='store_true', default=True)
parser.add_argument('--output', help="Output file name", default='output')
parser.add_argument('--min_df', help="Feature should be present in at least `min_df` instances (default 50)", default=50, type=int)
parser.add_argument('--iterations', help="number of models to fit (default 100)", default=100, type=int)

args = parser.parse_args()


mbtypes=["ENFJ", "ENTJ", "ENTP", "ESFJ", "ESFP", "ESTJ", "INFJ", "INFP", "INTJ", "INTP", "ISFJ", "ISTJ", "ISTP"]


def read_txt_data_file(datafile, metafile=None):
    meta_features=[]
    features=[]
    labels=[]

    if metafile is not None:
        meta_features = map(unicode.strip, codecs.open(metafile, encoding="utf-8").readlines())
        print len(meta_features), "meta features"

    for line_no, line in enumerate(codecs.open(datafile,encoding="utf-8")):
        if line and len(line.strip().split("\t")) == 4:
            # print line_no
            label, gender, tweetcount, text = line.strip().split("\t", 4)
            if args.dim is None:
                labels.append(label)
            else:
                labels.append(label[args.dim])

            f = Featurizer()
            f.word_ngrams(text, ngram="1-2-3")
            f.get_gender(gender)
            f.add_feature("tweetcount",tweetcount)

            if metafile is not None:
                try:
                    f.get_meta_features(meta_features[line_no])
                except IndexError:
                    print line_no, line
                    sys.exit()

            features.append(f.getDict())
        else:
            print >> sys.stderr, line_no, line
            print >> sys.stderr, "_____"

    assert(len(features)==len(labels))
    print Counter(labels)
    return features, labels


def show_most_informative_features(vectorizer, clf, n=10):
    feature_names = vectorizer.get_feature_names()
    for i in range(0,len(clf.coef_)):
        coefs_with_fns = sorted(zip(clf.coef_[i], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        print "i",i
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

####### load data
inputd, labels = read_txt_data_file(args.data, metafile=args.meta)


print >>sys.stderr, "{} instances loaded.".format(len(inputd))

np.random.seed(args.seed)

data ={}

labEnc = LabelEncoder()
labEnc.fit(labels)
print >> sys.stderr, "num labels: {}".format(len(labEnc.classes_))
print >> sys.stderr, "labels: {}".format(labEnc.classes_)


# target labels
data['target'] = labEnc.transform(labels)
data['DESC'] = "data set"
data['target_names'] = labEnc.classes_

# from text to features... (by default word n-grams) stored in sparse format
vectorizer = DictVectorizer()
data['data'] = vectorizer.fit_transform(inputd)
#print vectorizer.vocabulary_


allgold=[]
allpred=[]
enclabels=data['target']
X=data['data']


if not args.coef:
    clf = LogisticRegression(class_weight='auto')
    print(clf)

    cv = cross_validation.cross_val_score(clf, X, y=enclabels, cv=args.folds, n_jobs=10, verbose=1)

    majority_label = Counter(labels).most_common()[0][0]
    maj = [majority_label for x in xrange(len(labels))]
    rand = [random.sample(labels, 1)[0] for x in labels]

    print "classifier:", cv.std(), cv.mean()
    print "majority base:",accuracy_score(labEnc.transform(labels), labEnc.transform(maj))
    print "random base:",accuracy_score(labEnc.transform(labels), labEnc.transform(rand))


if args.coef:
    # Output
    file_basename = args.output

    sel = RandomizedLogisticRegression(n_jobs=10, n_resampling=args.iterations, sample_fraction=0.75, verbose=2)
    new_X = sel.fit_transform(X, enclabels)

    clf = LogisticRegression(class_weight='auto')
    clf.fit(new_X, enclabels)

    # this one does not get the probs
    # selected_feature_names = np.asarray(vectorizer.get_feature_names())[np.flatnonzero(clf.coef_[0])]
    # selected_feature_probs = clf.coef_[0][np.flatnonzero(clf.coef_[0])]

    # this one gets probs, but introduces a mismatch
    # selected_feature_names = np.asarray(vectorizer.get_feature_names())[np.flatnonzero(sel.scores_)]
    # selected_feature_probs = sel.scores_[np.flatnonzero(sel.scores_)]

    # this one works, it seems
    active_feature_mask = sel.get_support()
    selected_feature_names = np.asarray(vectorizer.get_feature_names())[active_feature_mask]
    selected_feature_probs = sel.scores_[active_feature_mask]

    print "%s instances and features, %s selected names, %s selected probability scores" % (new_X.shape, selected_feature_names.shape, selected_feature_probs.shape)
    print "%s coefficients" % (clf.coef_[0].shape)

    selection_probs = pd.DataFrame({'coefficients':clf.coef_[0], 'probabilities':selected_feature_probs}, index=selected_feature_names)
    selection_probs['abs'] = selection_probs.coefficients.abs()
    # selection_probs = selection_probs[(selection_probs != 0.0)]
    selection_probs.sort('abs', inplace=True)

    selection_probs.to_csv(os.path.join(file_basename + ".%s.stability-coef.csv" % ('ALL' if args.dim is None else args.dim)), sep='\t', encoding='utf-8')

    # selection_probs2 = pd.Series(sel.scores_, index=vectorizer.get_feature_names())
    # # selection_probs2 = selection_probs2[(selection_probs2 != 0.0)]
    # selection_probs2.sort()
    # selection_probs2.to_csv(os.path.join(file_basename + ".%s.stability-weight.csv" % ('ALL' if args.dim is None else args.dim)), sep='\t', encoding='utf-8')

