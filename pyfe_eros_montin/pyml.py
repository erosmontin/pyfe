VERSION="0.0.3"
import multiprocessing
from pynico_eros_montin import pynico as pn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
from sklearn.feature_selection import f_oneway

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
        self.trainingSplit=0.8
        
        self.validationReplicas=10
        self.ml=KNeighborsClassifier(3)
        self.fs=f_oneway
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
                d.write('feature,p,mean Healthy,mean MS\n')
                for f in self.getFeaturesName(s=a):
                    p,m0,m1=evaluateFeatures2(f0[f].to_numpy().squeeze(),f1[f].to_numpy().squeeze())
                    d.write(f'{f},{p},{m0},{m1}\n')
            d.close()
            P.undo()



    def __calc__(self):
        return testDataACC

    def calculate(self):
        while self.Subsets.size()>0:
            
            a=self.Subsets.pop()
            self.tmpSubsets.push(a)
            print(f"calculating subset {a['name']}")
            P=[]
            
            splits=splitPandasDataset(self.X.iloc[:,a["indexes"]],self.Y,train_ratio=self.trainingSplit,n=self.validationReplicas)
            for split in splits:
                xtr,ytr, xte,yte = split
                ml=deepcopy(self.ml)
                NR=self.trainingReplicas
                name=a["name"]
                other={"cv":5,"n_jobs":1}
                P.append([xtr,ytr,ml,NR,xte,yte,name,other])
            
            # classification10(xtr,ytr,ml,NR,xte,yte,name) #DEBUG
            with multiprocessing.Pool() as pp:
                O=pp.starmap(self.__calc__(),P)

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

def splitPandasDatasetv1(dataX,dataY,train_ratio = 0.75,n=1):
    GROUPS=groupingDA(dataX)
    O=[]
    for tr,te in GroupShuffleSplit(n_splits=n,train_size=train_ratio).split(dataX,dataY,groups=GROUPS):
        O.append([dataX.iloc[tr],dataY.iloc[tr],dataX.iloc[te],dataY.iloc[te]])
    if n==1:
        return O[0]
    else:
        return O
    
def splitPandasDataset(dataX,dataY,train_ratio = 0.75,n=1):
    GROUPS=groupingDA(dataX)
    O=[]
    
    for tr,te in StratifiedGroupKFold(n_splits=n,shuffle=True).split(dataX,dataY,groups=GROUPS):
        O.append([dataX.iloc[tr],dataY.iloc[tr],dataX.iloc[te],dataY.iloc[te]])
    if n==1:
        return O[0]
    else:
        return O




# def splitPandasDatasetRegression(dataX,dataY,train_ratio = 0.75):
#     # train is now 75% of the entire data set
#     # the _junk suffix means that we drop that variable completely
#     tmpX=deepcopy(dataX)
#     tmpY=deepcopy(dataY)
#     l=dataX.filter(like='aug',axis=0).index
#     tmpX=tmpX.drop(l)
#     tmpY=tmpY.drop(l)
#     x_train, x_test, y_train, y_test = train_test_split(tmpX, tmpY, test_size=1 - train_ratio)

#     L=x_train.index
#     LT=x_test.index

#     if len(l)>0: # if the dataset is augmnted
#         for a in L:
#             x=dataX.filter(like=a+'-aug',axis=0)
#             y=dataY.filter(like=a+'-aug',axis=0)
#             x_train=pd.concat([x_train,x])
#             y_train=pd.concat([y_train,y])
        
#         for a in LT:
#             x=dataX.filter(like=a+'-aug',axis=0)
#             y=dataY.filter(like=a+'-aug',axis=0)
#             x_test=pd.concat([x_test,x])
#             y_test=pd.concat([y_test,y])


#     return x_train,y_train, x_test, y_test



from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np
# sfs=SequentialFeatureSelector(KNeighborsClassifier(3), n_features_to_select=6)
from sklearn.preprocessing import StandardScaler

def sfs(X,Y,training_split=0.8,NF=6,NR=5,direction="backward",ml=KNeighborsClassifier(3),n_jobs=2,ncv=20,scale=False):
    K={}
    P=[]
    #prepare the structure for the multi process
    for _ in range(NR):
        tx,ty,*_=splitPandasDataset(X,Y,train_ratio=training_split)
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

def selectSubsetClassif(x,y,thetype=f_oneway,train_ratio=0.75,nr=10,n=None,index=None):
    K={}

    for aa in range(nr):
        tx,ty,*_=splitPandasDataset(x,y,train_ratio=train_ratio)
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
def testDataACC(X,Y,ml = None,replicas=100,Xval=None,Yval=None,name=None,other=None):
    out={"test":{"tn":[], "tp":[], "fp":[], "fn":[],"sensitivity":[],"specificity":[],"accuracy":[], "auc":[]},
        "validation":{"tn":[], "tp":[], "fp":[], "fn":[],"sensitivity":[],"specificity":[],"accuracy":[], "auc":[]},
        "model":None,"model_number":[],"features":list(X.columns),"name":name}
  
    if(Xval is not None):
        VAL=True


    accOut=0
    for pp in range(replicas):
        Xtr,Ytr,Xte,Ytest=splitPandasDataset(X.copy(),Y.copy(),0.75)
        
        SC=StandardScaler()
        SC.fit(Xtr)
        Xtr=pd.DataFrame(SC.transform(Xtr))
        Xte=pd.DataFrame(StandardScaler().fit_transform(Xte))
        Xte=Xte.fillna(0)
        Xtr=Xtr.fillna(0)
        ml.fit(Xtr.to_numpy(), Ytr.to_numpy().flatten())
        Yhat=ml.predict(Xte.to_numpy())
        try:
            ot=testPrediction(Ytest.to_numpy().flatten(),Yhat.flatten())
            tacc=ot["accuracy"]
            out["test"]["accuracy"].append(ot["accuracy"])
            out["test"]["sensitivity"].append(ot["sensitivity"])
            out["test"]["specificity"].append(ot["specificity"])
            out["test"]["tn"].append(ot["tn"])
            out["test"]["tp"].append(ot["tp"])
            out["test"]["fp"].append(ot["fp"])
            out["test"]["fn"].append(ot["fn"])
            out["test"]["auc"].append(ot["auc"])            
            if tacc>accOut+1e-5:
                accOut=tacc
                out["model"]=deepcopy(ml)
                out["model_number"].append(pp)
                out["modelscaler"]=SC


                if VAL:
                    try:
                        Xva=pd.DataFrame(SC.transform(Xval))
                        Xva=Xva.fillna(0)
                        Yhval=out["model"].predict(Xva.to_numpy())
                        ot =testPrediction(Yval.to_numpy().flatten(), Yhval.flatten())
                        out["validation"]["accuracy"].append(ot["accuracy"])
                        out["validation"]["sensitivity"].append(ot["sensitivity"])
                        out["validation"]["specificity"].append(ot["specificity"])
                        out["validation"]["tn"].append(ot["tn"])
                        out["validation"]["tp"].append(ot["tp"])
                        out["validation"]["fp"].append(ot["fp"])
                        out["validation"]["fn"].append(ot["fn"])
                        out["validation"]["auc"].append(ot["auc"])
            
                    except:
                        pass

            
        except:
            pass

    return out
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
        Xtr,Ytr,Xte,Ytest=splitPandasDataset(X.copy(),Y.copy(),0.75,n=1)
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

    # def calculate(self):
    #     while (self.Subsets.size()>0):
            
    #         a=self.Subsets.pop()
    #         self.tmpSubsets.push(a)
    #         print(f"calculating subset {a['name']}")
    #         P=[a]*self.validationReplicas
    #         with multiprocessing.Pool() as pp:
    #             O=pp.map(self.__calc__,P)
    #         for i,on in enumerate(O):
    #             if i==0:
    #                 o=on
    #                 o["model"]=[o["model"]]
    #             else:
    #                 o["validation"]["error"]+=on["validation"]["error"]
    #                 o["validation"]["r2"]+=on["validation"]["r2"]
    #                 o["validation"]["score"]+=on["validation"]["score"]
    #                 o["model"].append(on["model"])
                                
    #         L=pn.Pathable(self.resultFile)
    #         L.changeBaseName(f"{a['name']}.pkl")
    #         L.writePkl([o])
    #         H=[np.mean(o["validation"]["error"]), np.mean(o["validation"]["r2"]),np.mean(o["validation"]["score"]),len(o["features"])]
    #         p=pd.DataFrame.from_dict({o["name"]:H},orient='index')
    #         p.columns=["error", "r2","score","nfeatures"]
    #         self.SubsetsResults=pd.concat([self.SubsetsResults,p])
    # def writeSubsetsFeaturesName(self,n=None):
    #     if n==None:
    #         n=0
    #     P=pn.Pathable(self.resultFile)
    #     if self.Subsets.size()>0:
    #         tmp=deepcopy(self.Subsets)
    #     else:
    #         tmp=deepcopy(self.tmpSubsets)
        
    #     f1=self.X.filter(self.Y[self.Y.iloc[:,n]==1].index,axis=0)
    #     f0=self.X.filter(self.Y[self.Y.iloc[:,n]==0].index,axis=0)
    #     while (tmp.size()>0):
    #         a=tmp.pop()
    #         with open(P.changeBaseName(f'feartures{a["name"]}.txt').getPosition(),'w') as d:
    #             d.write('feature,p,mean Healthy,mean MS\n')
    #             for f in self.getFeaturesName(s=a):
    #                 p,m0,m1=evaluateFeatures2(f0[f].to_numpy().squeeze(),f1[f].to_numpy().squeeze())
    #                 d.write(f'{f},{p},{m0},{m1}\n')
    #         d.close()
    #         P.undo()
    # def addSubset(self,name,cols=None,icols=None,like=None,reducedSubset=True):
    #     super().addSubset(self,name,cols=None,icols=None,like=None,reducedSubset=True)

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
        "validation":{"tn":[], "tp":[], "fp":[], "fn":[],"sensitivity":[],"specificity":[],"accuracy":[], "auc":[]},
        "model":None,"model_number":[],"features":list(X.columns),"name":name}
    _X=X.interpolate()
    _X=_X.fillna(0)
    cv=20
    if 'cv' in other.keys():
        cv=other['cv']
    n_jobs=2
    if 'n_jobs' in other.keys():
        n_jobs=other['n_jobs']
    bestacc=0.0
    GROUPS=groupingDA(_X)
    skf = GroupShuffleSplit(n_splits=cv,test_size=0.2)
    for pp in range(replicas):
        #make a scikit pipeline
        clf = make_pipeline(StandardScaler(), ml)
        #cross validate

        test_acc=cross_validate(clf, _X.to_numpy(), Y.to_numpy().flatten(), cv=skf,n_jobs=n_jobs,return_estimator=True,groups=GROUPS)
        #select the most accurate set
        bestmodel=np.argmax(np.array(test_acc['test_score']))
        out["test"]["accuracy"].append(test_acc['test_score'])
        if out["test"]["accuracy"][-1][bestmodel]>bestacc:
            bestacc=out["test"]["accuracy"][-1][bestmodel]
            Ypred=test_acc['estimator'][bestmodel].predict(Xval)
            ot =testPrediction(Yval.to_numpy().flatten(), Ypred.flatten())
            out["validation"]["accuracy"].append(ot["accuracy"])
            out["validation"]["sensitivity"].append(ot["sensitivity"])
            out["validation"]["specificity"].append(ot["specificity"])
            out["validation"]["tn"].append(ot["tn"])
            out["validation"]["tp"].append(ot["tp"])
            out["validation"]["fp"].append(ot["fp"])
            out["validation"]["fn"].append(ot["fn"])
            out["validation"]["auc"].append(ot["auc"])
            out['model']=test_acc['estimator'][int(bestacc)]
    return out

if __name__=="__main__":


    L=Learner()
    L.setX('/data/MYDATA/NEWDATASET_TDCS/radiomic2/BB4/dataframeXaug.json')
    
    for t in range(1,9):
       L.X.drop(L.X.filter(like=f'aug00{t}',axis=0).index,inplace=True)

    MS=[]
    for t in L.X.index:
        if 'MS' in t:
            y=1
        else:
            y=0
        MS.append(y)

    Y=pd.DataFrame(MS,index=L.X.index)
    X=L.X
    Xtr,Ytr,Xte,Ytest=splitPandasDataset(X,
    Y,0.75)
    ml=KNeighborsClassifier(3)
    clf = make_pipeline(StandardScaler(), ml)
    cv=5
    SI05t=pml.getNFeaturesWithIportance(X,Y,n=5,scale=True)
    acc=cross_validate(clf, Xtr.iloc[:,SI05t], Ytr, cv=cv,n_jobs=10,return_estimator=True)
    print('f')
    # L=Regressor()
    # L.ml=LinearRegression()
    # L.setX('/data/MYDATA/ISMRM23/Aging/dataframeX_2.json')
    # PP=pn.Pathable('Aging_Study.txt')
    # J=pn.Pathable('/data/MYDATA/ISMRM23/Aging/Aging_Study.txt')
    # K=J.readJson()
    # Y = pd.DataFrame(columns=["age"],index=L.X.index)
    # for t in K:
    #     Y.loc[t['name']]=t['age']


    # L.selectKBestNumber=30
    # L.selectKBestReplicas=10
    # L.trainingReplicas=30
    # L.validationReplicas=5
    # L.trainingSplit=0.75
    # L.setY(Y)
    # L.addSubsetFull('ALL')

    # L.addSubset(like="T1",name="T1")
    # L.addSubset(like="T2",name="T2")
    # L.addSubset(like="FOS",name="FOS")
    # L.addSubset(like="SS",name="SS")
    # L.addSubset(like="Caudate",name="Caudate")
    # L.addSubset(like="Hippocampus",name="Hippocampus")
    # L.addSubset(like="Putamen",name="Putamen")

    # P= pn.Pathable('results.csv')
    # L.resultFile=P.getPosition()
    # L.calculate()


    # L.saveResults()
   

# from pynico_eros_montin import pynico as pn
# L=pn.createTemporaryPathableFromFileName("t.pkl")
# o={"aa":9}
# L.writePkl(o)


    import pandas as pd
    from pynico_eros_montin import pynico as pn
  
    from sklearn.neighbors import KNeighborsClassifier
  

    def create_list_of_strings(Max,Min=0):
        list_of_strings = []
        for i in range(Max, Min, -1):
            list_of_strings.append("aug" + str(i).zfill(4))
        return list_of_strings

    from copy import deepcopy
    class myLerner(Learner):
        def __init__(self, x=None, y=None) -> None:
            super().__init__(x, y)
        
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
                    d.write('feature,p,value 0, value 1\n')
                    for f in self.getFeaturesName(s=a):
                        try:
                            p,m0,m1=evaluateFeatures2(f0[f].to_numpy().squeeze(),f1[f].to_numpy().squeeze())
                        except:
                            print("AAA")
                        d.write(f'{f},{p},{m0},{m1}\n')
                d.close()
                P.undo()



    LL=KNeighborsClassifier(3)
    #LL=DecisionTreeClassifier(max_depth=5)
    #LL=RandomForestClassifier(max_depth=5,random_state=0),
    # LL=AdaBoostClassifier()
    # LL=SVC(kernel="linear")
    # LL= GaussianNB()
    # LL= QuadraticDiscriminantAnalysis()
    # LL=  SVC()
    # LL=GaussianProcessClassifier()



    # PT='/data/PROJECTS/marcoMS/RadiomicsAppliedtoPhaseContrast/link_NEWDATASET_TDCS/radiomic/Man'
    # for i in range(90, -1, -10):
    #     GAD=myLerner()
    #     list_of_strings = create_list_of_strings(Max=100,Min=i)
    #     P= pn.Pathable(f'{PT}/results_K3{100-len(list_of_strings):04d}/results.csv')
    #     GAD.setX(f'{PT}/dataframeXaug.json')
    #     GAD.X=GAD.X.sort_index()
    #     MS=[]
    #     for t in GAD.X.index:
            
    #         if 'MS' in t:
    #             y=1
 
    #         else:
    #             y=0
    #         MS.append(y)

    #     Y=pd.DataFrame(MS,index=GAD.X.index)
    #     GAD.Y=Y
    #     # removing some instances
    #     for aa in list_of_strings:
    #         GAD.Y.drop(GAD.Y.filter(like=aa,axis=0).index,inplace=True)
    #         GAD.X.drop(GAD.X.filter(like=aa,axis=0).index,inplace=True)


    #     GAD.selectKBestNumber=3
    #     GAD.selectKBestReplicas=10
    #     GAD.trainingReplicas=100
    #     GAD.validationReplicas=5
    #     GAD.selectKBesttrainsplit=0.9


    #     GAD.ml=LL

    #     P.ensureDirectoryExistence()
    #     GAD.resultFile=P.getPosition()
    #     GAD.addSubset(like="FOS.Signal",name="FOSSignal")
    #     GAD.calculate()
    #     GAD.saveResults()
    #     GAD.writeSubsetsFeaturesName()
