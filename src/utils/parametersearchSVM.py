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



def dowloadEmbedding(BERT,AlBERTo,BERT3,BERT3_Un,MultiBERT,BERTCased,Electra,TypeEmb): 
    seed_everything(1234)
    embeddings_method=['cls_last_hidden_state','last_hidden_state_average','last_hidden_state_concat','four_last_hidden_state_concat','four_last_hidden_state_sum']

    train_df = pd.read_csv('data/processed/sentipolcAlBERTo/TrainAlBERToSentiPolc.csv', converters={'rep': literal_eval, 'spans': literal_eval})
    test_df = pd.read_csv('data/processed/sentipolcAlBERTo/TestAlBERToSentiPolc.csv')
    if(BERT):
        bert_rep=BertRep("MiBo/RepML")
        embeddings_method_BERT = getattr(bert_rep, embeddings_method[TypeEmb])
        train_df['embBERT'] = train_df['Stralci_predetti'].map(embeddings_method_BERT).values.tolist()
        test_df['embBERT'] = test_df['Stralci_predetti'].map(embeddings_method_BERT).values.tolist()

    if(AlBERTo):
        AlBERTo_rep = BertRep("Marco127/DRAlBERTo")
        embeddings_method_AlBERTo = getattr(AlBERTo_rep, embeddings_method[TypeEmb])
        train_df['embAlBERT'] = train_df['Stralci_predetti'].map(embeddings_method_AlBERTo).values.tolist()
        test_df['embAlBERT'] = test_df['Stralci_predetti'].map(embeddings_method_AlBERTo).values.tolist()

    
    if(BERT3):
        bert_rep_3 = BertRep("Marco127/BERTUnCasedColor")
        embeddings_method_BERT_3 = getattr(bert_rep_3, embeddings_method[TypeEmb])
        train_df['embBERT_3'] = train_df['Stralci_predetti'].map(embeddings_method_BERT_3).values.tolist()
        test_df['embBERT_3'] = test_df['Stralci_predetti'].map(embeddings_method_BERT_3).values.tolist()
    
    if(BERT3_Un):
        bert_rep_3_unbalanced= BertRep("Marco127/AlBERToColorUnbalanced")
        embeddings_method_BERT_3_Unbalced = getattr(bert_rep_3_unbalanced, embeddings_method[TypeEmb])
        train_df['embBERT_3_Un'] = train_df['Stralci_predetti'].map(embeddings_method_BERT_3_Unbalced).values.tolist()
        test_df['embBERT_3_Un'] = test_df['Stralci_predetti'].map(embeddings_method_BERT_3_Unbalced).values.tolist()
    
    if(MultiBERT):
        multiBERT_rep= BertRep("Marco127/MultiBert")
        embeddings_method_MultiBERT = getattr(multiBERT_rep, embeddings_method[TypeEmb])
        train_df['embMultiBERT'] = train_df['Stralci_predetti'].map(embeddings_method_MultiBERT).values.tolist()
        test_df['embMultiBERT'] = test_df['Stralci_predetti'].map(embeddings_method_MultiBERT).values.tolist()
    if(BERTCased):
        CasedBERT_rep= BertRep("Marco127/BERTCased")
        embeddings_method_BERTCased = getattr(CasedBERT_rep, embeddings_method[TypeEmb])
        train_df['embCasedBERT'] = train_df['Stralci_predetti'].map(embeddings_method_BERTCased).values.tolist()
        test_df['embCasedBERT'] = test_df['Stralci_predetti'].map(embeddings_method_BERTCased).values.tolist()
    if(Electra):
        Electra_rep= BertRep("Marco127/Electra")
        embeddings_method_Electra = getattr(Electra_rep, embeddings_method[TypeEmb])
        train_df['embElectra'] = train_df['Stralci_predetti'].map(embeddings_method_Electra).values.tolist()
        test_df['embElectra'] = test_df['Stralci_predetti'].map(embeddings_method_Electra).values.tolist()

def modelComposition(A,B,C,D,ModelA,ModelB,ModelC,ModelD,Task):
    X_train=[[((i*A)+(j*B)+(z*C)+(D*h))/(A+B+C+D) for i, j, z, h in zip(x, y , o, k)] for x, y, o, k  in zip( train_df["emb"+ModelA].to_list(),train_df["emb"+ModelB].to_list(),train_df["emb"+ModelC].to_list(),train_df["emb"+ModelD].to_list())]
    X_test=[[((i*A)+(j*B)+(z*C)+(D*h))/(A+B+C+D) for i, j, z, h in zip(x, y, o, k)] for x, y, o, k in zip(test_df["emb"+ModelA].to_list(),test_df["emb"+ModelB].to_list(),test_df["emb"+ModelC].to_list(),test_df["emb"+ModelD].to_list())]
    y_train=train_df[Task].to_list()
    y_test=test_df[Task].to_list()
    return X_train,X_test,y_train,y_test

def black_box_function1(C,gamma):
    # C: SVC hyper parameter to optimize for.
    model = SVC(C = C, gamma=gamma, class_weight="balanced")
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)
    f = f1_score(y_test, y_score, average='macro')+f1_score(y_test, y_score, pos_label=0)+f1_score(y_test, y_score, pos_label=1)
    #f=f1_score(y_test, y_score, pos_label=0)
    return f
def black_box_function2(C):
    # C: SVC hyper parameter to optimize for.
    model = SVC(C = C, gamma='scale', class_weight="balanced")
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)
    f = f1_score(y_test, y_score, average='macro')+f1_score(y_test, y_score, pos_label=0)+f1_score(y_test, y_score, pos_label=1)
    #f=f1_score(y_test, y_score, pos_label=0)
    return f


def SupportVectorMachine(class_weight,C,kernel,gamma):
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


def CrossValidation(model, cv, scoring):   
    scores = cross_val_score(
      model, X_train, y_train, cv=5, scoring='f1_macro')
    print("------------")

    print('Punteggio per ogni subset')
    print(scores)
    print("------------")

    print('Media')
    print(sum(scores)/len(scores))
    
    return sum(scores)/len(scores),scores


def SupportVectorMachineValidation(class_weight, C, kernel,gamma):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=45)
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