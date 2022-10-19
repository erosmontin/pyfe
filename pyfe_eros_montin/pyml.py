from pynico_eros_montin import pynico as pn
import pandas as pd
class Learner:
    def __init__(self,x=None,y=None) -> None:
        if x:
            self.X=x
        else:
            self.X=pd.DataFrame()

        if y:
            self.Y=y
        else:
            self.Y=pd.DataFrame()

        self.Xsubsets=pn.Stack()
        self.Ysubsets=pn.Stack()

    def setXfromJson(self,x,indexes,level=2):
        P=pn.Pathable(x)
        I=pn.Pathable(indexes)
        if P.exists():
            self.X=pd.json_normalize(P.readJson(),max_level=level)
        if I.exists():
            self.X.index=I.readJson()
        self.X.sort_index(axis=0,inplace=True,ascending=False)

    def setYfromJson(self,x,indexes,level=2):
        P=pn.Pathable(x)
        if P.exists():
            self.X=pd.json_normalize(P.readJson(),max_level=level)
        I=pn.Pathable(indexes)
        if I.exists():
            self.Y.index=I.readJson()
        self.Y.sort_index(axis=0,inplace=True,ascending=False)
    def addXSubset(self,name,col=None,icols=None,like=None):
        SS={"name":name}
        if icols:
            SS["indexes"]=icols

        self.Xsubsets.push(SS)

        

    

        
import os

if __name__=="__main__":
    L=Learner()
    L.setXfromJson('/data/tttt/a.json','/data/tttt/ai.json',2)
    print(L.X.index)
    L.addSubset(like=".SS.",name="ShapeAndSize")
