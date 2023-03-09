from copy import deepcopy
import ctypes
import json

import os
import numpy
from pynico_eros_montin import pynico as pn
from pydaug_eros_montin import pydaug as pda

class FE():
    def __init__(self,image=None,roi=None,roiv=None,PT=None,options=None) -> None:
        self.info={
            "type":"origin",
            "author":"eros.montin@gmail.com",
            "git":"",
        }

  
        self.PT=PT        
        self.Options={}
        self.Image=None
        self.ROI=None
        self.ROIvalue=1
        if image:
            self.setImage(image)
        if roi:
            self.setROI(roi)
        if roiv:
            self.setROIvalue(roiv)
        if options:
            self.setOptions(options)
    
    def getOptions(self):
        return self.Options
    def setImage(self,s):
        self.Image=s

    def getImage(self):
        return self.Image

    def setROI(self,s):
        self.ROI=s

    def getROI(self):
        return self.ROI

    def setROIvalue(self,s):
        self.ROIvalue=s

    def getROIvalue(self):
        return self.ROIvalue

    def getFeatures(self):
        pass
    
    def setOptions(self, o=None):
        if o:
            for key in self.Options.keys():
                try: 
                    if key in o.keys():               
                        self.Options[key]=o[key]
                except:
                    pass
    def writeoptionsTofile(self,filename):
        if filename:
            P=pn.Pathable(filename)
        else:
            P=pn.createRandomTemporaryPathableFromFileName('a.json')
        P.writeJson(self.Options)



    
    
     
class BD2DecideFE(FE):
    def __init__(self,dimension=3,image=None,roi=None,roiv=None,PT=None,options=None) -> None:
        super().__init__(image,roi,roiv,PT)
        self.trash=pn.GarbageCollector()

        self.Options["min"]="N"
        self.Options["nt"]=12
        self.Options["max"]="N"
        self.Options["bin"]=32
        self.Options["marginalScale"]=0.5
        self.outputN= pn.createRandomTemporaryPathableFromFileName('a.txt',self.PT).getPosition()
        self.outputV= pn.createRandomTemporaryPathableFromFileName('a.txt',self.PT).getPosition()
        self.trash.throw(self.outputN)
        self.trash.throw(self.outputV)
        self.dimension=dimension
        if options:
            self.setOptions(options)
    def __getResults__(self,domain=None):
        F=[]
        out={}
        with open(self.outputV) as f:
            while line := f.readline():
                F.append(line)
        c=0
        with open(self.outputN) as n:
            while line2 := n.readline():
                out[line2.strip()]=float(F[c].strip())
                c+=1
        if domain:
            out={domain:out}
        return out
    def __execute__(self,dll,args,domain=None):
        handle = ctypes.CDLL(dll)
        handle.main.argtypes =[ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]
        handle.main(len(args),args)
        if ((pn.Pathable(self.outputN).exists()) and (pn.Pathable(self.outputN).exists())):
            out= self.__getResults__(domain)
        return out
    
        

class SS(BD2DecideFE):

    def getFeatures(self,moredomain=None):
        if not moredomain:
            moredomain=''
        aa="N"
        basepath = os.path.dirname(os.path.abspath(__file__))
        args=(ctypes.c_char_p*7)(b'ss',self.ROI.encode(),str(self.ROIvalue).encode(),aa.encode(),self.outputV.encode(),self.outputN.encode(),str(self.Options["nt"]).encode())
        out =self.__execute__(os.path.join(basepath,"bld/libfe3dss.so"),args,moredomain + "SS")
        return out

class FOS(BD2DecideFE):
    def getFeatures(self,moredomain=None):
        MASK=''
        if not moredomain:
            moredomain=''
  
        basepath = os.path.dirname(os.path.abspath(__file__))
        args=(ctypes.c_char_p*10)(b'fos',self.Image.encode(),self.ROI.encode(),str(self.ROIvalue).encode(),
        str(self.Options["min"]).encode(),str(self.Options["max"]).encode(),
        str(self.Options["bin"]).encode(),str(self.Options["marginalScale"]).encode(),
        self.outputV.encode(),self.outputN.encode())
        out={}
        if self.dimension==3:
            out =self.__execute__(os.path.join(basepath,"bld/libfe3dfos.so"),args,moredomain + "FOS")
        elif self.dimension==2:
            out =self.__execute__(os.path.join(basepath,"bld/libfe2dfos.so"),args,moredomain + "FOS")

        return out

class TEXTURES(BD2DecideFE):
    def __init__(self, image=None, roi=None, roiv=None, PT=None,options=None) -> None:
        super().__init__(image, roi, roiv, PT)
        self.Options["radius"]=1
        if options:
            self.setOptions(options)

class GLCM(TEXTURES):
     def getFeatures(self,moredomain=None):

        if not moredomain:
            moredomain=''
        
     
        basepath = os.path.dirname(os.path.abspath(__file__))
        args=(ctypes.c_char_p*11)(b'glcm',self.Image.encode(),self.ROI.encode(),str(self.ROIvalue).encode(),
        str(self.Options["min"]).encode(),str(self.Options["max"]).encode(),
        str(self.Options["bin"]).encode(),str(self.Options["radius"]).encode(),
        self.outputV.encode(),self.outputN.encode(),
        str(self.Options["nt"]).encode())


        out={}
        if self.dimension==3:
            out =self.__execute__(os.path.join(basepath,"bld/libfe3dglcm.so"),args,moredomain + "GLCM_" + str(self.Options["radius"]))
        elif self.dimension==2:
            out =self.__execute__(os.path.join(basepath,"bld/libfe2dglcm.so"),args,moredomain + "GLCM_" + str(self.Options["radius"]))

        return out


class GLRLM(TEXTURES):
     def getFeatures(self,moredomain=None):

        if not moredomain:
            moredomain=''

        basepath = os.path.dirname(os.path.abspath(__file__))
        args=(ctypes.c_char_p*11)(b'glrlm',self.Image.encode(),self.ROI.encode(),str(self.ROIvalue).encode(),
        str(self.Options["min"]).encode(),str(self.Options["max"]).encode(),
        str(self.Options["bin"]).encode(),str(self.Options["radius"]).encode(),
        self.outputV.encode(),self.outputN.encode(),
        str(self.Options["nt"]).encode())
        
        out={}
        if self.dimension==3:
            out =self.__execute__(os.path.join(basepath,"bld/libfe3dglrlm.so"),args,moredomain + "GLRLM_" + str(self.Options["radius"]))
        elif self.dimension==2:
            out =self.__execute__(os.path.join(basepath,"bld/libfe2dglrlm.so"),args,moredomain + "GLRLM_" + str(self.Options["radius"]))

        return out
        

import multiprocessing
def exrtactMyFeatures(jf,dimension,daug=0):
    if isinstance(jf,str):
        P=pn.Pathable(jf)
        if not P.exists():
            raise Exception("file do not exists")
        L=P.readJson()
    elif (isinstance(jf,dict)):
        L=jf
    if dimension==3:
        f=theF3
    if dimension==2:
        f=theF2

    with multiprocessing.Pool() as p:
        # result = p.map(theF3,(L["dataset"]))
        res = p.map(f,(L["dataset"]))
    p.close()
    result=[]
    idx=[]
       

    for r,id in res:
        if (isinstance(id,list)):
            for rr,idr in zip(r,id):
                result.append(rr)
                idx.append(idr)
        else:
            result.append(r)
            idx.append(id)
    return result,idx

import pandas as pd
def exrtactMyFeaturesToPandas(jf,dimension,max_level=2,daug=0):
    r,ind=exrtactMyFeatures(jf,dimension,daug)
    X=pd.json_normalize(r,max_level=max_level)
    X.index=ind
    return X

def pyfejsonFeaturesToPandas(xf,idf):
    P=pn.Pathable(xf)
    I=pn.Pathable(idf)
    if P.exists():
        X=pd.json_normalize(P.readJson(),max_level=2)
        X.index=pd.json_normalize(I.readJson(),max_level=2)
    return X

def theF3(X):
    return theF(X,3)
def theF2(X):
    return theF(X,2)
from pyable_eros_montin import imaginable
import numpy as np
import SimpleITK as sitk
def theF(X,d):
    line=X["data"]
    r=[]
    ids=[]
    r.append(computeRow(line,d))
    ids.append(X["id"])

    if "augment" in X.keys():
        aug=X["augment"]
        for n in range(aug["n"]):

            R=[ np.random.uniform(low=l, high=h, size=1)[0] for l,h in aug["options"]["r"]]
            T=[ np.random.uniform(low=l, high=h, size=1)[0] for l,h in aug["options"]["t"]]
            line2 =deepcopy(line)
            G=pn.GarbageCollector()
            for i,x in enumerate(line2):
                im,roi=x["image"],x["labelmap"]
                
                IM=imaginable.SITKImaginable(filename=im)
                ROI=imaginable.SITKImaginable(filename=roi)
                IM.rotateImage(rotation=R,translation=T,interpolator=sitk.sitkBSpline)
                ROI.rotateImage(rotation=R,translation=T,useNearestNeighborExtrapolator=True,interpolator=sitk.sitkNearestNeighbor)

                pn.Pathable(im)
                imn=pn.Pathable(im).changePathToOSTemporary().changeFileNameRandom()
                roin=pn.Pathable(roi).changePathToOSTemporary().changeFileNameRandom()
                G.throw(imn.getPosition())
                G.throw(roin.getPosition())
                IM.writeImageAs(imn.getPosition())
                ROI.writeImageAs(roin.getPosition())
                line2[i]["image"]=imn.getPosition()
                line2[i]["labelmap"]=roin.getPosition()
            r.append(computeRow(line2,d))

            ids.append(f'{X["id"]}-aug{n:04}')




    return r,ids

    
    
def computeRow(line,d):
    out={}
    for x in line:
        TD=[s["name"].lower() for s in x["groups"]]
        TDs=[s["options"] for s in x["groups"]]
        for a,o in zip(TD,TDs):
            if a=="ss":
                L=SS(d)
            if a=="fos":
                L=FOS(d)
            if a=="glcm":
                L=GLCM(d)
            if a=="glrlm":
                L=GLRLM(d)

            L.setImage(x["image"])
            L.setROI(x["labelmap"])

            L.setROIvalue(x["labelmapvalue"])
            L.setOptions(o)
            f=L.getFeatures()
            ft=list(f.keys())[0]
            if not (x["groupPrefix"] in out.keys()):
                out[x["groupPrefix"]]={}
            if not (ft in out[x["groupPrefix"]].keys()):
                out[x["groupPrefix"]][ft]={}
            out[x["groupPrefix"]][ft]= f[ft]
        

    return out


if __name__=="__main__":
    # P=pn.Pathable('/data/PERSONALPROJECTS/myPackages/pyfe/tests/data/data.json')
    # P=pn.Pathable('/data/MYDATA/TDCS/EROS_TDCS/_Hman.json')
    # o=P.readJson()
    # x=o["dataset"][0][0]



    # L=GLCM(3)
    # L.setImage(x["image"])
    # L.setROI(x['labelmap'])
    # L.setROIvalue(x["labelmapvalue"])
    # L.setOptions(x["settings"])
    # L.getFeatures()
    # print(L.getFeatures())
#    o,i=exrtactMyFeatures('/data/MYDATA/TDCS/EROS_TDCS/_Hman.json',2)

    # P=pn.Pathable('/data/MYDATA/TDCS/EROS_TDCS/Hman.json')

    # o=P.readJson()
    # x=o["dataset"][0]["data"][5]



    # L=GLCM(3)
    # L.setImage(x["image"])
    # L.setROI(x['labelmap'])
    # L.setROIvalue(x["labelmapvalue"])
    # L.setOptions(x["groups"][0]["options"])
    # L.getFeatures()
    # print(L.getFeatures())
 
    # MDj,dimension='/data/PERSONALPROJECTS/myPackages/pyfe/tests/data/data.json',3
    # MDj,dimension='/data/MYDATA/TDCS/EROS_TDCS/radiomic/Man/Hman.json',2
    MDj,dimension='/g/a.json',3
 #  o,i=exrtactMyFeatures('/data/PERSONALPROJECTS/myPackages/pyfe/tests/data/data.json',3)
    # P=pn.Pathable(MDj)
    # o=P.readJson()
    # x=o["dataset"][0]["data"][5]
    # L=FOS(2)
    # L.setImage(x["image"])
    # L.setROI(x['labelmap'])
    # L.setROIvalue(x["labelmapvalue"])
    # L.setOptions(x["groups"][0]["options"])
    # L.getFeatures()

    p=exrtactMyFeaturesToPandas(MDj,dimension)
    p.to_json('/R.json')

#    print(i)
#    P=pn.Pathable('/data/tttt/a.json')
#    P.writeJson(o)
#    P=pn.Pathable('/data/tttt/ai.json')
#    P.writeJson(i)



    # A=GLCM(3)
    # print(A.getOptions())
    # A.setImage('/data/MYDATA/AUGMENTED_RADIOMIC_HIP3/input/Leftp02_rt_0.149_0.247_-6.060_2.163_-4.351_3.036_noise_0.nii.gz')
    # # A.setImage(x["image"])
    # A.min=0
    # A.max=512
    # A.setROI('/data/MYDATA/AUGMENTED_RADIOMIC_HIP3/output/Leftp02_rt_0.149_0.247_-6.060_2.163_-4.351_3.036_noise_0.nii.gz')
    # # A.setROI('/data/MYDATA/AUGMENTED_RADIOMIC_HIP3/output/Leftp02_rt_0.149_0.247_-6.060_2.163_-4.351_3.036_noise_0.nii.gz')
    # # A.setROI(x["labelmap"])
    # # A.setROIvalue(x["labelmapvalue"])
    # A.setROIvalue(1)
    # A.setOptions({})
    # print(A.getFeatures())
    # # A=GLCM(3)
    # # A.setImage('/data/t.nii.gz')
    # # A.max="N"
    # # A.min="N"
    # # A.setROI('/data/t.nii.gz')
    # # print(A.getFeatures())