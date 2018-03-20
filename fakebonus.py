# Imports
import numpy as np

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from build_sets import *

def DecisionTrees(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label):
    clf = tree.DecisionTreeClassifier(max_depth=150)
    clf = clf.fit(training_set_np, training_label)

    print("Decision Trees")
    print("Training Set Accuracy  : " + str(100*clf.score(training_set_np, training_label)))
    print("Validation Set Accuracy: " + str(100*clf.score(validation_set_np, validation_label)))
    print("Testing Set Accuracy   : " + str(100*clf.score(testing_set_np, testing_label)))
    print("\n")

def RandomForest(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label):
    clf = RandomForestClassifier(max_depth=128, n_estimators=256)
    clf = clf.fit(training_set_np, training_label)

    print("Random Forest")
    print("Training Set Accuracy  : " + str(100*clf.score(training_set_np, training_label)))
    print("Validation Set Accuracy: " + str(100*clf.score(validation_set_np, validation_label)))
    print("Testing Set Accuracy   : " + str(100*clf.score(testing_set_np, testing_label)))
    print("\n")

def MultinomialNB_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label):
    clf = MultinomialNB()
    clf.fit(training_set_np, training_label)

    print("MultinomialNB")
    print("Training Set Accuracy  : " + str(100*clf.score(training_set_np, training_label)))
    print("Validation Set Accuracy: " + str(100*clf.score(validation_set_np, validation_label)))
    print("Testing Set Accuracy   : " + str(100*clf.score(testing_set_np, testing_label)))
    print("\n")

def GaussianNB_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label):
    clf = GaussianNB()
    clf.fit(training_set_np, training_label)

    print("GaussianNB")
    print("Training Set Accuracy  : " + str(100*clf.score(training_set_np, training_label)))
    print("Validation Set Accuracy: " + str(100*clf.score(validation_set_np, validation_label)))
    print("Testing Set Accuracy   : " + str(100*clf.score(testing_set_np, testing_label)))
    print("\n")

def MLP_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label):
    clf = MLPClassifier(hidden_layer_sizes=[2048, 1024, 512, 256])
    clf.fit(training_set_np, training_label)

    print("MLPClassifier")
    print("Training Set Accuracy  : " + str(100*clf.score(training_set_np, training_label)))
    print("Validation Set Accuracy: " + str(100*clf.score(validation_set_np, validation_label)))
    print("Testing Set Accuracy   : " + str(100*clf.score(testing_set_np, testing_label)))
    print("\n")

def MLP_clf__KBest(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label):
    selector = SelectKBest(score_func=f_classif, k=1600)
    selector.fit(training_set_np, training_label)
    training_set_np = selector.transform(training_set_np)
    validation_set_np = selector.transform(validation_set_np)
    testing_set_np = selector.transform(testing_set_np)

    clf = MLPClassifier(hidden_layer_sizes=[2048, 1024, 512, 256])
    clf.fit(training_set_np, training_label)

    print("MLP Classifier K Best")
    print("Training Set Accuracy  : " + str(100*clf.score(training_set_np, training_label)))
    print("Validation Set Accuracy: " + str(100*clf.score(validation_set_np, validation_label)))
    print("Testing Set Accuracy   : " + str(100*clf.score(testing_set_np, testing_label)))
    print("\n")

def PassiveAggressive_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label):
    clf = PassiveAggressiveClassifier(max_iter = 150)
    clf.fit(training_set_np, training_label)

    print("Passive Aggressive Classifier")
    print("Training Set Accuracy  : " + str(100*clf.score(training_set_np, training_label)))
    print("Validation Set Accuracy: " + str(100*clf.score(validation_set_np, validation_label)))
    print("Testing Set Accuracy   : " + str(100*clf.score(testing_set_np, testing_label)))
    print("\n")

################################################################################
################################################################################
################################################################################

training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label = build_set_tfidf()
DecisionTrees(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
RandomForest(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
MultinomialNB_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
# GaussianNB_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
MLP_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
MLP_clf__KBest(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
PassiveAggressive_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)

training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label = build_set_count()
DecisionTrees(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
RandomForest(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
MultinomialNB_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
# GaussianNB_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
MLP_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
MLP_clf__KBest(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
PassiveAggressive_clf(training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label)
