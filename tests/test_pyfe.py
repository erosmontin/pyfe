
from pyfe_eros_montin import pyfe as pf
from pynico_eros_montin import pynico as pn

import multiprocessing
def fetchMyData(jf):
    P=pn.Pathable(jf)
    if not P.exists():
        raise Exception("file do not exists")
    L=P.readJson()

    with multiprocessing.Pool() as p:
        # result = p.map(theF3,(L["dataset"]))
        result = p.map(theF3,(L["dataset"]))
    return result



def theF3(X,analysis=None):
    if not analysis:
        analysis=["ss","fos","glcm","glrlm"]
    TD=[]
    analysis=[a.lower() for a in analysis]

    for a in analysis:
        if a=="ss":
            TD.append(pf.SS(3))
        if a=="fos":
            TD.append(pf.FOS(3))
        if a=="glcm":
            TD.append(pf.GLCM(3))
        if a=="glrlm":
            TD.append(pf.GLRLM(3))


    out={}
    for x in X:
        for L in TD:
            L.setImage(x["image"])
            L.setROI(x["labelmap"])
            L.setROIvalue(x["labelmapvalue"])
            L.setOptions(x["settings"])
            out.update(L.getFeatures(x["name"]))
    return out
 
import os

F=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/data.json')
o=fetchMyData(F)
P=pn.createRandomTemporaryPathableFromFileName('a.json','/data/tttt')
P.writeJson(o)