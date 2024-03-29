import toml

# with open('pyproject.toml', 'r') as f:
#     config = toml.load(f)
 
# # Access values from the config
# VERSION=config['project']['version']
import multiprocessing
from pynico_eros_montin import pynico as pn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
from sklearn.feature_selection import f_oneway
import matplotlib.pyplot as plt

def setPandasDataFrame(fn):
    if (isinstance(fn,str)):
        if pn.Pathable(fn).exists():
            X=pd.read_json(fn)
    elif(isinstance(fn,pd.DataFrame)):
        X=fn
    return X

class Learner:
    def __init__(self,x=None,y=None) -> None:
        self.selectKBestReplicas=10
        self.selectKBestNumber=4
        self.selectKBesttrainsplit=0.8
        self.trainingReplicas=100
        self.cv=5
        self.trainingSplit=0.8
        self.className=['0','1']
        self.validationReplicas=10
        self.ml=KNeighborsClassifier(3)
        self.fs=f_oneway
        self.parallel=True
        self.splitfunction=splitPandasDatasetStratifiedGroupKFold
        if x:
            self.X=x
        else:
            self.X=pd.DataFrame()

        if y:
            self.Y=y
        else:
            self.Y=pd.DataFrame()
        self.resultFile=pn.createTemporaryPosition('O.pandas')
        self.Subsets=pn.Stack()
        self.tmpSubsets=pn.Stack()
        self.SubsetsResults=pd.DataFrame(columns=["accuracy","sensitivity","specificity","auc","nfeatures"])
    def reset(self):
        pass

    def setXfromPyfeJson(self,x,indexes,level=2):
        P=pn.Pathable(x)
        I=pn.Pathable(indexes)
        if P.exists():
            self.X=pd.json_normalize(P.readJson(),max_level=level)
        if I.exists():
            self.X.index=I.readJson()
        
    def setX(self,fn):
        self.X=setPandasDataFrame(fn)
        self.reset()

    def appendRows(self,fn):
        X=setPandasDataFrame(fn)
        self.X = self.X.append(X)

    
    def setY(self,fn):
        if (isinstance(fn,str)):
            if pn.Pathable(fn).exists():
                self.Y=pd.read_json(fn)
                self.reset()
        elif(isinstance(fn,pd.DataFrame)):
            self.Y=fn
            self.reset()
     

    def getFeaturesName(self,s=None):
        if s==None:
            s=self.getSubset()
        return self.X.columns[s["indexes"]]

    def saveResults(self):
        self.SubsetsResults.to_csv(self.resultFile)


    def evaluateFeatures2(self,x,y,bplt=None):
        p,m0,m1=evaluateFeatures2(x,y)
        
        if bplt is not None:
            fig=plt.figure()
            plt.boxplot([x,y],labels=self.className,patch_artist=True)
            plt.title(f'{bplt["title"]} (P:{p:0.2e})')
            plt.xlabel('feature')
            plt.ylabel('feature values')
            plt.grid()
            if 'fn' in bplt.keys():
                if not 'dpi' in bplt.keys():
                    dpi=100
                    if bplt['fn'] is not None:
                        plt.savefig(bplt['fn'].replace('.','_'))

            return p,m0,m1,fig

        
        return p,m0,m1
        
            
        
    def writeSubsetsFeaturesName(self,n=None):
        if n==None:
            n=0
        P=pn.Pathable(self.resultFile)
        if self.Subsets.size()>0:
            tmp=deepcopy(self.Subsets)
        else:
            tmp=deepcopy(self.tmpSubsets)
        
        f1=self.X.filter(self.Y[self.Y.iloc[:,n]==1].index,axis=0)
        f0=self.X.filter(self.Y[self.Y.iloc[:,n]==0].index,axis=0)
        while (tmp.size()>0):
            a=tmp.pop()
            with open(P.changeBaseName(f'feartures{a["name"]}.txt').getPosition(),'w') as d:
                d.write(f'Feature,p,{self.classNames[0]},{self.classNames[1]}\n')
                for f in self.getFeaturesName(s=a):
                    p,m0,m1=self.evaluateFeatures2(f0[f].to_numpy().squeeze(),f1[f].to_numpy().squeeze())
                    d.write(f'{f},{p},{m0},{m1}\n')
            d.close()
            P.undo()
    def getVariableInTheFinalModel(self,resultfile):
        a=pn.readPkl(resultfile)
        return a[0][0]['original_results']['features']
    
    def checkResultsFeatures(self,resultfile,OUT=None,conf=None):
        
        f1=self.Y[self.Y.iloc[:,0]==1].index
        f0=self.Y[self.Y.iloc[:,0]==0].index
        O=[]
        features=self.getVaraibleInThefinalmodel(resultfile)
        for f in features:
            X0=self.X.loc[f0,f]
            X1=self.X.loc[f1,f]
            if OUT is not None:
                out=f'{OUT}{f}.png'
                pn.Pathable(out).ensureDirectoryExistence()
            else:
                out=None
            p,m0,m1,fig=self.evaluateFeatures2(X0.to_numpy().squeeze(),X1.to_numpy().squeeze(),{'title':f,'fn':out})
            O.append([p,m0,m1,fig])
        return O


    def __calc__(self):
        return classification10

    def calculate(self):
        while self.Subsets.size()>0:
            
            a=self.Subsets.pop()
            self.tmpSubsets.push(a)
            print(f"calculating subset {a['name']}")
            P=[]
            SPF=self.splitfunction
            splits=SPF(self.X.iloc[:,a["indexes"]],self.Y,train_ratio=self.trainingSplit,n=self.validationReplicas)
            for split in splits:
                xtr,ytr, xte,yte = split
                ml=deepcopy(self.ml)
                NR=self.trainingReplicas
                name=a["name"]
                other={"cv":5,"n_jobs":1,"splitfunction":self.splitfunction}
                P.append([xtr,ytr,ml,NR,xte,yte,name,other])
            
            if self.parallel:
                with multiprocessing.Pool() as pp:
                    O=pp.starmap(self.__calc__(),P)
                pp.close()
            else:
                O=[]
                f=self.__calc__()
                for l in P:

                    o=f(*l)
                    O.append(o)
            p={}
            # print('debug','here')
            for o in O:
                for k in o.keys():
                    if not k in p.keys():
                        p[k]={}
                    if isinstance(o[k],dict):
                        for k2 in o[k].keys():
                            if not k2 in p[k].keys():
                                p[k][k2]=[]
                            p[k][k2].append(o[k][k2][-1])
                    else:
                        p[k]=o[k]             
            METRICS=p['validation'].keys()

            L=pn.Pathable(self.resultFile)
            L.changeBaseName(f"{a['name']}.pkl")
            L.writePkl([{"original_results":o,"last_results":p}])
            H=[*[np.mean(p['validation'][m]) for m in METRICS],*[np.std(p['validation'][m]) for m in METRICS]]
            H+=[len(o["features"])]
            dataframe=pd.DataFrame.from_dict({p["name"]:H},orient='index')
            HN=[*[ 'mean_' +m for m in METRICS],*['std_' + m for m in METRICS],'nfeatures']
            dataframe.columns=HN
            if len(self.SubsetsResults)>0:
                self.SubsetsResults=pd.concat([self.SubsetsResults,dataframe])
            else:
                self.SubsetsResults=dataframe


    def getSubset(self):
        return self.Subsets.peek()
    def addSubsetFull(self,name,common=False):
        try:
            self.addSubset(name,cols=list(self.X.columns),common=common)
        except:
            raise Exception("can't add 'All' subset!!")


    def addSubset(self,name,cols=None,icols=None,like=None,reducedSubset=True,common=False):
        

        SS={"name":name,"indexes":[]}
        if icols:
            if isinstance(icols,int):
                icols=[icols]

            for u in icols:
                SS["indexes"].append(u)
        if cols:
            if isinstance(cols,str):
                cols=[cols]
            for u in cols:
                try:
                    SS["indexes"].append(self.X.columns.get_loc(u))
                except:
                    pass
        if like:
            if isinstance(like,str):
                like=[like]
            
            for l in like:
                try:
                    LL=[self.X.columns.get_loc(c) for c in list(self.X.filter(like=l).columns)]
                except:
                    pass
                for l2 in LL:
                    SS["indexes"].append(l2)
                    
        
        if (len(SS["indexes"])>0):
            if common:
                try:
                    LL=[self.X.columns.get_loc(c) for c in list(self.X.filter(like="common").columns)]
                except:
                    pass
                for l2 in LL:
                    SS["indexes"].append(l2)

            self.Subsets.push(SS)
            if reducedSubset:
                
                # nn=selectSubsetClassif(self.X.iloc[:,SS["indexes"]],self.Y,train_ratio=self.selectKBesttrainsplit,nr=self.selectKBestReplicas,n=self.selectKBestNumber,thetype=self.fs)
                nn=getNFeaturesWithImportance(self.X.iloc[:,SS["indexes"]],self.Y,self.selectKBestNumber,scale=False)
                SS2={"indexes":[SS["indexes"][x] for x in nn],"name":f"{name}_subset{self.selectKBestNumber}"}
                if (len(SS2["indexes"])>0):
                    if common:
                        try:
                            LL=[self.X.columns.get_loc(c) for c in list(self.X.filter(like="common").columns)]
                        except:
                            pass
                        for l2 in LL:
                            SS2["indexes"].append(l2)
                    self.Subsets.push(SS2)


from copy import deepcopy             

from sklearn.model_selection import train_test_split
def splitPandasDatasetv0(dataX,dataY,train_ratio = 0.75):
    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    tmpX=deepcopy(dataX)
    tmpY=deepcopy(dataY)
    l=dataX.filter(like='aug',axis=0).index
    tmpX=tmpX.drop(l)
    tmpY=tmpY.drop(l)
    x_train, x_test, y_train, y_test = train_test_split(tmpX, tmpY, test_size=1 - train_ratio)

    L=x_train.index
    LT=x_test.index

    if len(l)>0: # if the dataset is augmnted
        for a in L:
            x=dataX.filter(like=a+'-aug',axis=0)
            y=dataY.filter(like=a+'-aug',axis=0)
            x_train=pd.concat([x_train,x])
            y_train=pd.concat([y_train,y])
        
        for a in LT:
            x=dataX.filter(like=a+'-aug',axis=0)
            y=dataY.filter(like=a+'-aug',axis=0)
            x_test=pd.concat([x_test,x])
            y_test=pd.concat([y_test,y])


    return x_train,y_train, x_test, y_test
from sklearn.model_selection import GroupShuffleSplit,GroupKFold,StratifiedGroupKFold

def splitPandasDatasetGroupShuffleSplit(dataX,dataY,train_ratio = 0.75,n=1):
    GROUPS=groupingDA(dataX)
    O=[]
    for tr,te in GroupShuffleSplit(n_splits=n,train_size=train_ratio).split(dataX,dataY,groups=GROUPS):
        O.append([dataX.iloc[tr],dataY.iloc[tr],dataX.iloc[te],dataY.iloc[te]])
    if n==1:
        return O[0]
    else:
        return O
    
def splitPandasDatasetStratifiedGroupKFold(dataX,dataY,train_ratio = 0.75,n=1):
    GROUPS=groupingDA(dataX)
    O=[]    
    for tr,te in StratifiedGroupKFold(n_splits=n,shuffle=True).split(dataX,dataY,groups=GROUPS):
        O.append([dataX.iloc[tr],dataY.iloc[tr],dataX.iloc[te],dataY.iloc[te]])
    if n==1:
        return O[0]
    else:
        return O





from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np
# sfs=SequentialFeatureSelector(KNeighborsClassifier(3), n_features_to_select=6)
from sklearn.preprocessing import StandardScaler

def sfs(X,Y,training_split=0.8,NF=6,NR=5,direction="backward",ml=KNeighborsClassifier(3),n_jobs=2,ncv=20,scale=False,splitfunction=splitPandasDatasetGroupShuffleSplit):
    K={}
    P=[]
    #prepare the structure for the multi process
    for _ in range(NR):
        tx,ty,*_=splitfunction(X,Y,train_ratio=training_split)
        if scale:
            Xva=pd.DataFrame(StandardScaler().fit_transform(tx))
        else:    
            Xva=tx
        Xva=Xva.interpolate()
        Xva=Xva.fillna(0)
        sfs=SequentialFeatureSelector(ml, n_features_to_select=NF,direction=direction,n_jobs=n_jobs,cv=ncv)
        P.append([Xva.to_numpy(), ty.to_numpy().squeeze(),sfs])

    pool = multiprocessing.Pool()
    results = pool.starmap(_fit_sffs, P)

    pool.close()
    pool.join()
    #identify the most selected features
    for f in results:
        for _f in f:
            if _f in K.keys():
                K[_f]+=1
            else:
                K[_f]=1
    L= [ list(K.keys())[i] for i in np.array(list(K.values())).argsort()[::-1]]
    if NF:
        return L[0:NF]
    else:
        return L




from sklearn.ensemble import RandomForestClassifier



def _fit_sffs(x,y,sfs):
    fit = sfs.fit(x,y)
    # np.where returns an array
    return list(np.where(fit.get_support())[0])

from sklearn.ensemble import RandomForestClassifier

def featuresImportance(X,Y,scale=False):
    forest = RandomForestClassifier(random_state=0)
    if scale:
        X=StandardScaler().fit_transform(X)
    if isinstance(Y,pd.DataFrame):
        Y=Y.to_numpy().ravel()
    forest.fit(X, Y)
    return forest.feature_importances_

def rankFeaturesImportance(X,Y,scale=False,method='GINI'):
    importance=featuresImportance(X,Y,scale)
    return sorted(zip(importance,range(len(importance))), key=lambda x: x[0], reverse=True)

def getNFeaturesWithImportance(X,Y,n,scale=False):
    F = rankFeaturesImportance(X,Y,scale)
    O=[]
    for t in range(n):
        O.append(F[t][-1])
    return O


def forwardSequentialFeatureSelector(X,Y,training_split=0.8,NF=6,NR=5,ml=KNeighborsClassifier(3),ncv=10,scale=False):
    return sfs(X,Y,training_split=training_split,NF=NF,NR=NR,direction="forward",ml=ml,ncv=ncv,n_jobs=1,scale=scale)

def backwardSequentialFeatureSelector(X,Y,training_split=0.8,NF=6,NR=5,ml=KNeighborsClassifier(3),ncv=10,scale=False):
    return sfs(X,Y,training_split=training_split,NF=NF,NR=NR,direction="backward",ml=ml,ncv=ncv,n_jobs=1,scale=scale)

def forwardSequentialFeatureSelectorNoReplicas(X,Y,NF=6,ml=KNeighborsClassifier(3),ncv=10,scale=False,n_jobs=10):
    return forwardBackwardSequentialFeatureSelectorNoReplicas(X,Y,NF=NF,ml=ml,ncv=ncv,scale=scale,n_jobs=n_jobs,direction='forward')

def backwardSequentialFeatureSelectorNoReplicas(X,Y,NF=6,ml=KNeighborsClassifier(3),ncv=10,scale=False,n_jobs=10):
    return forwardBackwardSequentialFeatureSelectorNoReplicas(X,Y,NF=NF,ml=ml,ncv=ncv,scale=scale,n_jobs=n_jobs,direction='backward')


def forwardBackwardSequentialFeatureSelectorNoReplicas(X,Y,NF=6,ml=KNeighborsClassifier(3),ncv=10,scale=False,n_jobs=10,direction='backward'):
        if scale:
            _X=pd.DataFrame(StandardScaler().fit_transform(X).copy())
        else:    
            _X=X
        X.interpolate()
        _X=_X.fillna(0)
        sfs=SequentialFeatureSelector(ml, n_features_to_select=NF,direction=direction,n_jobs=n_jobs,cv=ncv)
        return _fit_sffs(_X.to_numpy(), Y.to_numpy().squeeze(),sfs)
 

from sklearn.feature_selection import SelectKBest

import numpy as np

def selectSubsetClassif(x,y,thetype=f_oneway,train_ratio=0.75,nr=10,n=None,index=None,splitfunction=splitPandasDatasetGroupShuffleSplit):
    K={}

    for aa in range(nr):
        tx,ty,*_=splitfunction(x,y,train_ratio=train_ratio)
        Xva=pd.DataFrame(StandardScaler().fit_transform(tx))
        Xva=Xva.interpolate()
        Xva=Xva.fillna(0)

        if n:
            test = SelectKBest(score_func=thetype,k=n)
        else:
            test = SelectKBest(score_func=thetype)
        
        fit = test.fit(Xva.to_numpy(), ty.to_numpy().squeeze())
        features = fit.transform(np.linspace(0,len(tx.columns)-1,len(tx.columns),dtype=int).reshape(1,-1))
        for f in features[0]:
            if f in K.keys():
                K[f]+=1
            else:
                K[f]=1

    L= [ list(K.keys())[i] for i in np.array(list(K.values())).argsort()[::-1]]
    
    if n:
        return L[0:n]
    else:
        return L


from scipy.stats import ranksums

def evaluateFeatures2(f0,f1):
    _,p=ranksums(f0,f1)
    return p, f0.mean(), f1.mean()




from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
eps=1e-6
def testPrediction(Ygt,Yhat):
    tn_, fp_, fn_, tp_ = confusion_matrix(Ygt.flatten(), Yhat.flatten()).ravel()
    tp_=float(tp_)
    tn_=float(tn_)
    fn_=float(fn_)
    fp_=float(fp_)
    try:
        auc=roc_auc_score(Ygt.flatten(), Yhat.flatten())
    except:
        auc=np.nan
    tacc=(tp_+tn_)/(tp_+tn_+fn_+fp_+eps)
    o={"accuracy":tacc,
    "specificity":( tn_ /(tn_+fp_+eps)),
    "sensitivity":( tp_ /(tp_+fn_+eps)),
    "tn":tn_,
    "tp":tp_,
    "fp":fp_,
    "fn":fn_,
    "auc":auc}
    return o

from sklearn.metrics import confusion_matrix as CM

from pynico_eros_montin import pynico as pn
# Read the Excel file into a Pandas DataFrame
# from xgboost import XGBClassifier
from scipy.stats.contingency import relative_risk as r_f
from scipy.stats.contingency import odds_ratio as o_f
import sklearn.metrics as metrics

eps=1e-6
def binaryPredictionEvaluation(gt,test):
  
  """
  Calculates all the statistical measurements from a confusion matrix.

  Args:
    gt is the ground truth
    test is the test values

  Returns:
    A dictionary of statistical measurements.
  """
  confusion_matrix=np.float32(CM(gt,test))
  # Get the number of data points.
  n = confusion_matrix.sum()

  # Get the true positives, false positives, false negatives, and true negatives.
  true_positives = confusion_matrix[0, 0]
  false_positives = confusion_matrix[0, 1]
  false_negatives = confusion_matrix[1, 0]
  true_negatives = confusion_matrix[1, 1]

  # Calculate the accuracy.
  accuracy = (true_positives + true_negatives) / n

  # Calculate the precision.
  precision = true_positives / (true_positives + false_positives+eps)

  # Calculate the recall.
  recall = true_positives / (true_positives + false_negatives+eps)

  # Calculate the F1 score.
  f1 = 2 * (precision * recall) / (precision + recall+eps)

  # Calculate the specificity.
  specificity = true_negatives / (true_negatives + false_positives+eps)

  # Calculate the Matthews correlation coefficient (MCC).
  mcc = (true_positives * true_negatives - false_positives * false_negatives) / (
      np.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives))+eps)

  # Calculate the error rate.
  error_rate = (false_positives + false_negatives) / n
 
 # Calculate the relative risk.
  
#   A=[a if a>0 else 1 for a in A]
  try:
    relative_risk = r_f(*list(confusion_matrix.astype(int).ravel())).relative_risk
    if np.isinf(relative_risk):
        raise Exception()
  except:
    relative_risk =np.nan

  # Calculate the odds ratio.
#   if ((false_negatives==0) or (true_negatives==0)):
  try:
    odds_ratio = o_f(list(confusion_matrix.astype(int))).statistic
    if np.isinf(relative_risk):
        raise Exception()
  except:
    odds_ratio =np.nan


#   else:
    #   odds_ratio =np.nan

  # Calculate the sensitivity.
  sensitivity = true_positives / ((true_positives + false_negatives)+eps)

  # Calculate the specificity.
  specificity = true_negatives / ((true_negatives + false_positives)+eps)

  # Calculate the AUC.
  auc = metrics.roc_auc_score(gt,test)

  # Calculate the AUC threshold.
  fpr, tpr, thresholds=metrics.roc_curve(gt,test)
  optimal_idx = np.argmax(tpr - fpr)
  auc_threshold = thresholds[optimal_idx]
  

  # Return 
  # Return the dictionary of statistical measurements.
  return {
      "true_negatives": true_negatives/n,
      "true_positives": true_positives/n,
      "false_negatives": false_negatives/n,
      "false_positives": false_positives/n,

      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1": f1,
      "specificity": specificity,
      "mcc": mcc,
      "error_rate": error_rate,
            "relative_risk": relative_risk,
      "odds_ratio": odds_ratio,
            "sensitivity": sensitivity,
      "specificity": specificity,
      "auc": auc,
      "auc_threshold": auc_threshold,
  }


from sklearn.metrics import mean_squared_error, r2_score
def testRegressorACC(X,Y,ml = None,replicas=100,Xval=None,Yval=None,name=None,other=None):
    out={"test":{"error":[], "r2":[], "score":[]},
        "validation":{"error":[], "r2":[], "score":[]},
        "model":None,"model_number":[],"features":list(X.columns),"name":name}

  
    if(Xval is not None):
        VAL=True
        Xva=pd.DataFrame(StandardScaler().fit_transform(Xval).copy())
        Xva=Xva.fillna(0)

    errOut=np.inf
    for pp in range(replicas):
        Xtr,Ytr,Xte,Ytest=splitPandasDatasetGroupShuffleSplit(X.copy(),Y.copy(),0.75,n=1)
        Xtr=pd.DataFrame(StandardScaler().fit_transform(Xtr))
        Xte=pd.DataFrame(StandardScaler().fit_transform(Xte))
        Xte=Xte.fillna(0)
        Xtr=Xtr.fillna(0)
        ml.fit(Xtr.to_numpy(), Ytr.to_numpy().flatten())
        Yhat=ml.predict(Xte.to_numpy())

        try:
            score=ml.score(Xte.to_numpy(),Ytest.to_numpy().flatten())
            err=mean_squared_error(Ytest.to_numpy().flatten(),Yhat.flatten())
            r2=r2_score(Ytest.to_numpy().flatten(),Yhat.flatten())
            out["test"]["error"].append(err)
            out["test"]["r2"].append(r2)
            out["validation"]["score"].append(score)
            if err<errOut:
                errOut=err
                out["model"]=deepcopy(ml)
                out["model_number"].append(pp)


                if VAL:
                    try:
                        Yhval=out["model"].predict(Xva.to_numpy())
                        score=ml.score(Xva.to_numpy(),Yval.to_numpy().flatten())
                        err=mean_squared_error(Yval.to_numpy().flatten(), Yhval.flatten())
                        r2=r2_score(Yval.to_numpy().flatten(), Yhval.flatten())
                        
                        out["validation"]["error"].append(err)
                        out["validation"]["r2"].append(r2)
                        out["validation"]["score"].append(score)
            
                    except:
                        pass

            
        except:
            pass

    return out
class Regressor(Learner):
    def __init__(self, x=None, y=None) -> None:
        super().__init__(x, y)
        # self.SubsetsResults=pd.DataFrame(columns=["error","r2","score","nfeatures"])
    def __calc__(self):
        return testRegressorACC



class BinaryLearner(Learner):
    def __init__(self, x=None, y=None) -> None:
        super().__init__(x, y)
        # self.SubsetsResults=pd.DataFrame(columns=["error","r2","score","nfeatures"])
    def __calc__(self):
        return classification10


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
def groupingDA(X,daug='-aug'):
    #get the indexes without da
    UNIQUE=X.drop(X.filter(like=daug,axis=0).index).index.to_list()
    GROUPES=[]
    for a in X.index:
        n=a.find('-aug')
        if n>0:
            a=a[:n]
        GROUPES.append(UNIQUE.index(a))
    return GROUPES

from sklearn.model_selection import GroupShuffleSplit
def classification10(X,Y,ml = None,replicas=5,Xval=None,Yval=None,name=None,other={}):
    print((Y==0).sum(),(Y==1).sum())
    out={"test":{"accuracy":[]},
        "validation":{},
        "model":None,"model_number":[],"features":list(X.columns),"name":name}
    _X=X.interpolate()
    _X=_X.fillna(0)
    cv=20
    if 'cv' in other.keys():
        cv=other['cv']
    n_jobs=2
    if 'n_jobs' in other.keys():
        n_jobs=other['n_jobs']
    splitfunction=splitPandasDatasetStratifiedGroupKFold
    if 'splitfunction' in other.keys():
        splitfunction=other['splitgunciton']

    bestacc=-1
    GROUPS=groupingDA(_X)
    skf=splitfunction(n_splits=cv,shuffle=True)
    # skf = GroupShuffleSplit(n_splits=cv,test_size=0.2)
    for pp in range(replicas):
        #make a scikit pipeline
        clf = make_pipeline(StandardScaler(), ml)
        #cross validate
        test_acc=cross_validate(clf, _X.to_numpy(), Y.to_numpy().flatten(), cv=skf,n_jobs=n_jobs,return_estimator=True,groups=GROUPS)
        #select the most accurate model
        bestmodel=np.argmax(np.array(test_acc['test_score']))
        out["test"]["accuracy"].append(test_acc['test_score'])
        if out["test"]["accuracy"][-1][bestmodel]>bestacc:
            bestacc=out["test"]["accuracy"][-1][bestmodel]
            Ypred=test_acc['estimator'][bestmodel].predict(Xval)
            ot =binaryPredictionEvaluation(Yval.to_numpy().flatten(), Ypred.flatten())
            for k,v in ot.items():
                if not k in out['validation']:
                    out['validation'][k]=[]
                out['validation'][k].append(v)
            out['model']=test_acc['estimator'][int(bestacc)]
    return out

# if __name__=="__main__":
#     GAD=BinaryLearner()
#     GAD.setX('/data/MYDATA/ANO-INT/extraction_Max.json')

#     risposta=pd.read_csv('/data/MYDATA/ANO-INT/risposta.csv')

#     Y=pd.DataFrame()
#     X=pd.DataFrame()
#     GAD.X.interpolate(inplace=True)
#     GAD.X.fillna(0,inplace=True)

#     # filter the patients are in the study after the review
#     N1=1
#     N2=1
#     for pz,rc in zip(risposta['pz'],risposta['RC']):
        
#         index=GAD.X.filter(like=pz,axis=0).index
        
#         if rc==1:
#             index=index[0:N1] 
#         if rc==0:
#             index=index[0:(N2)]
#         print(len(index))

#         X=pd.concat((X,GAD.X.loc[index]))
#         Y=pd.concat((Y,pd.DataFrame([rc]*len(index),index=index)))
#     GAD.X=X
#     GAD.Y=Y
#     GAD.checkResultsFeatures('/data/MYDATA/ANO-INT/ECRresults/MAX/GINI20.pkl','/g/P/')
