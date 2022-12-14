import multiprocessing
from pynico_eros_montin import pynico as pn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy

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



    def __calc__(self,a):
        x=self.X.iloc[:,a["indexes"]]
        xtr,ytr, xte,yte = splitPandasDataset(x,self.Y,self.trainingSplit)
        return testDataACC(xtr,ytr,ml = self.ml,replicas=self.trainingReplicas,Xval=xte,Yval=yte,name=a["name"])
        
    def calculate(self):
        while (self.Subsets.size()>0):
            
            a=self.Subsets.pop()
            self.tmpSubsets.push(a)
            print(f"calculating subset {a['name']}")
            P=[a]*self.validationReplicas
            with multiprocessing.Pool() as pp:
                O=pp.map(self.__calc__,P)
            for i,on in enumerate(O):
                if i==0:
                    o=on
                    o["model"]=[o["model"]]
                else:
                    o["validation"]["sensitivity"]+=on["validation"]["sensitivity"]
                    o["validation"]["specificity"]+=on["validation"]["specificity"]
                    o["validation"]["auc"]+=on["validation"]["auc"]
                    o["validation"]["accuracy"]+=on["validation"]["accuracy"]
                    o["validation"]["accuracy"]+=on["validation"]["accuracy"]
                    o["model"].append(on["model"])
                                
            L=pn.Pathable(self.resultFile)
            L.changeBaseName(f"{a['name']}.pkl")
            L.writePkl([o])
            H=[np.mean(o["validation"]["sensitivity"]), np.mean(o["validation"]["specificity"]),np.mean(o["validation"]["accuracy"]),np.mean(o["validation"]["auc"]),len(o["features"])]
            p=pd.DataFrame.from_dict({o["name"]:H},orient='index')
            p.columns=["sensitivity", "specificity","accuracy","auc","nfeatures"]
            self.SubsetsResults=pd.concat([self.SubsetsResults,p])
    def getSubset(self):
        return self.Subsets.peek()
    def addSubsetFull(self,name):
        try:
            self.addSubset(name,cols=list(self.X.columns))
        except:
            raise Exception("can't add 'All' subset!!")


    def addSubset(self,name,cols=None,icols=None,like=None,reducedSubset=True):
        
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
            self.Subsets.push(SS)
            if reducedSubset:
                nn=selectSubsetClassif(self.X.iloc[:,SS["indexes"]],self.Y,train_ratio=self.selectKBesttrainsplit,nr=self.selectKBestReplicas,n=self.selectKBestNumber)
                SS2={"indexes":[SS["indexes"][x] for x in nn],"name":f"{name}_subset{self.selectKBestNumber}"}
                if (len(SS2["indexes"])>0):
                    self.Subsets.push(SS2)


from copy import deepcopy             

from sklearn.model_selection import train_test_split
def splitPandasDataset(dataX,dataY,train_ratio = 0.75):
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





from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_oneway

import numpy as np

def selectSubsetClassif(x,y,thetype=f_oneway,train_ratio=0.75,nr=10,n=None,index=None):
    K={}

    for aa in range(nr):
        tx,ty,*_=splitPandasDataset(x,y,train_ratio=train_ratio)
        Xva=pd.DataFrame(StandardScaler().fit_transform(tx).copy())
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

def testPrediction(Ygt,Yhat):
    tn_, fp_, fn_, tp_ = confusion_matrix(Ygt.flatten(), Yhat.flatten()).ravel()
    tacc=(tp_+tn_)/(tp_+tn_+fn_+fp_)
    o={"accuracy":tacc,
    "specificity":( tn_ /(tn_+fp_)),
    "sensitivity":( tp_ /(tp_+fn_)),
    "tn":tn_,
    "tp":tp_,
    "fp":fp_,
    "fn":fn_,
    "auc":roc_auc_score(Ygt.flatten(), Yhat.flatten())}
    return o
def testDataACC(X,Y,ml = None,replicas=100,Xval=None,Yval=None,name=None):
    out={"test":{"tn":[], "tp":[], "fp":[], "fn":[],"sensitivity":[],"specificity":[],"accuracy":[], "auc":[]},
        "validation":{"tn":[], "tp":[], "fp":[], "fn":[],"sensitivity":[],"specificity":[],"accuracy":[], "auc":[]},
        "model":None,"model_number":[],"features":list(X.columns),"name":name}

  
    if(Xval is not None):
        VAL=True
        Xva=pd.DataFrame(StandardScaler().fit_transform(Xval).copy())
        Xva=Xva.fillna(0)

    accOut=0
    for pp in range(replicas):
        Xtr,Ytr,Xte,Ytest=splitPandasDataset(X.copy(),Y.copy(),0.75)
        SC=StandardScaler()
        SC.fit(Xtr)
        Xtr=pd.DataFrame(StandardScaler().fit_transform(Xtr))
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
            if tacc>accOut:
                accOut=tacc
                out["model"]=deepcopy(ml)
                out["model_number"].append(pp)


                if VAL:
                    try:
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
if __name__=="__main__":
    L=Learner()
    # L.setXfromMyfeJson('/data/tttt/a.json','/data/tttt/ai.json',2)
    # print(L.X.index)
    PT='/data/MYDATA/TDCS/EROS_TDCS/radiomic/BB4/'
    L.setX(PT+'dataframeXaug.json')
    
    Y=pd.DataFrame([ 1 for a in  L.X.filter(like ='MS',axis=0).index], index=L.X.filter(like ='MS',axis=0).index,columns=["MS"])
    Y=pd.concat([Y,pd.DataFrame([ 0 for a in  L.X.filter(like ='NC',axis=0).index], index=L.X.filter(like ='NC',axis=0).index,columns=["MS"])])
    L.setY(Y)
    L.selectKBestNumber=10
    L.selectKBestReplicas=100
    L.trainingReplicas=100
    L.validationReplicas=20
    P= pn.Pathable(PT+'/results_10/results.csv')
    P.ensureDirectoryExistence()
    L.resultFile=P.getPosition()
    L.addSubset(like="FOS.Signal",name="FOSSignal")
    L.addSubset(like="FOS.Histogram",name="FOSHistogram")
    L.addSubset(like="FOS",name="FOS")
    L.addSubsetFull("All")
    L.addSubset(like="GLCM",name="GLCM")
    L.addSubset(like="GLRLM",name="GLRLM")
    L.addSubset(like="GL",name="Texture")
    L.addSubset(like="_lca.",name="LCA")
    L.addSubset(like="lca.FOS",name="LCAFOS")
    L.addSubset(like="lca.GLCM",name="LCAGLCM")
    L.addSubset(like="lca.GLRLM",name="LCAGLRLM")
    L.addSubset(like="lca.GL",name="LCATexture")
    L.addSubset(like="lva.FOS",name="LVAFOS")
    L.addSubset(like="lva.GLCM",name="LVAGLCM")
    L.addSubset(like="lva.GLRLM",name="LVAGLRLM")
    L.addSubset(like="lva.GL",name="LVATexture")
    L.addSubset(like="rca.FOS",name="RCAFOS")
    L.addSubset(like="rca.GLCM",name="RCAGLCM")
    L.addSubset(like="rca.GLRLM",name="RCAGLRLM")
    L.addSubset(like="rca.GL",name="RCAGTextureM")
    L.addSubset(like="rva.FOS",name="RVAFOS")
    L.addSubset(like="rva.GLCM",name="RVAGLCM")
    L.addSubset(like="rva.GLRLM",name="RVAGLRLM")
    L.addSubset(like="rva.GL",name="RVATexture")
    L.addSubset(like="_lva.",name="LVA")
    L.addSubset(like="_rca.",name="RCA")
    L.addSubset(like="_rva.",name="RVA")    
    L.addSubset(like="magnitude",name="magnitude")
    L.addSubset(like="phase",name="phase")
    L.addSubset(like="phase_lva",name="phase_lva")
    L.addSubset(like="phase_lca",name="phase_lca")
    L.addSubset(like="phase_rca",name="phase_rca")
    L.addSubset(like="phase_rva",name="phase_rva")
    L.addSubset(like="magnitude_lva",name="magnitude_lva")
    L.addSubset(like="magnitude_lca",name="magnitude_lca")
    L.addSubset(like="magnitude_rca",name="magnitude_rca")
    L.addSubset(like="magnitude_rva",name="magnitude_rva")


    print(L.calculate())

    L.saveResults()
    L.writeSubsetsFeaturesName()
   

# from pynico_eros_montin import pynico as pn
# L=pn.createTemporaryPathableFromFileName("t.pkl")
# o={"aa":9}
# L.writePkl(o)
 