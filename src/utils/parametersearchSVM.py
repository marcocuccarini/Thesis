import random
import pandas as pd
from src.utils.utils import seed_everything
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
import sklearn.metrics
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from src.models.bert_rep import BertRep



def black_box_function1(C,gamma,X_train,X_test,y_train,y_test):
    # C: SVC hyper parameter to optimize for.
    model = SVC(C = C, gamma=gamma, class_weight="balanced")
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)
    f = f1_score(y_test, y_score, average='macro')+f1_score(y_test, y_score, pos_label=0)+f1_score(y_test, y_score, pos_label=1)
    #f=f1_score(y_test, y_score, pos_label=0)
    return f

def black_box_function2(C,X_train,X_test,y_train,y_test):
    # C: SVC hyper parameter to optimize for.
    model = SVC(C = C, gamma='scale', class_weight="balanced")
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)
    f = f1_score(y_test, y_score, average='macro')+f1_score(y_test, y_score, pos_label=0)+f1_score(y_test, y_score, pos_label=1)
    #f=f1_score(y_test, y_score, pos_label=0)
    return f


def SupportVectorMachine(class_weight,C,kernel,gamma,X_train,X_test,y_train,y_test):
    clf = svm.SVC(class_weight=class_weight, C=C, kernel=kernel,gamma= gamma)
    clf.fit(X_train, y_train)
    y_pred_test=clf.predict(X_test)

    print("Test on test_set")
    f1_1 = f1_score(y_test, y_pred_test, pos_label=1)
    f1_0 = f1_score(y_test, y_pred_test, pos_label=0)
    f1_mean = f1_score(y_test, y_pred_test, average='macro')
    print("Label0"+" "+ str(f1_0))
    print("Label1"+" "+ str(f1_1))
    print("Macro"+" "+ str(f1_mean))
    return clf


def CrossValidation(model, cv, scoring,X_train,X_test,y_train,y_test):   
    scores = cross_val_score(
      model, X_train, y_train, cv=cv, scoring='f1_macro')
    print("------------")

    print('Punteggio per ogni subset')

    print(scores)
    print("------------")

    print('Media')
    print(sum(scores)/len(scores))
    
    return sum(scores)/len(scores),scores


def SupportVectorMachineValidation(class_weight, C, kernel,gamma,X_train,X_test,y_train,y_test):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=int(random.rondom()*200))
    clf = svm.SVC(class_weight=class_weight, C=C, kernel=kernel,gamma=gamma)
    clf.fit(X_train, y_train)
    y_pred_val=clf.predict(X_val)
    y_pred_test=clf.predict(X_test)
    print("Test on validation_set")
    f1_1 = f1_score(y_val, y_pred_val, pos_label=1)
    f1_0 = f1_score(y_val, y_pred_val, pos_label=0)
    f1_mean = f1_score(y_val, y_pred_val, average='macro')
    print("Label0"+" "+ str(f1_0))
    print("Label1"+" "+ str(f1_1))
    print("Macro"+" "+ str(f1_mean))

    print("Test on test_set")
    f1_1 = f1_score(y_test, y_pred_test, pos_label=1)
    f1_0 = f1_score(y_test, y_pred_test, pos_label=0)
    f1_mean = f1_score(y_test, y_pred_test, average='macro')
    print("Label0"+" "+ str(f1_0))
    print("Label1"+" "+ str(f1_1))
    print("Macro"+" "+ str(f1_mean))

def BayesianOpt(black_box_function,pbounds,random_state):
    optimizer = BayesianOptimization(f = black_box_function,
                                     pbounds = pbounds, 
                                     verbose = 2,
                                     random_state = random_state,
                                     allow_duplicate_points=True)
    optimizer.maximize(init_points = 5, n_iter = 10)
    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))