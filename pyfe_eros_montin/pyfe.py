from copy import deepcopy
import ctypes
import json

import os
import numpy
from pynico_eros_montin import pynico as pn

from pydaug_eros_montin import pydaug as pda

import pyable_eros_montin.dev as dev
import toml


# print(__file__)
# with open(f'{pn.Pathable(__file__).getPath()}/../pyproject.toml', 'r') as f:
# #     config = toml.load(f)
 
# # Access values from the config
# VERSION=config['project']['version']

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
import radiomics.featureextractor as prsfe
import pyable_eros_montin.imaginable as ima
class PYRAD(TEXTURES):
    def __init__(self, image=None, roi=None, roiv=None, PT=None, options=None) -> None:
        super().__init__(image, roi, roiv, PT, options)
        self.featurestype=['firstorder','shape','glcm','glrlm','glszm','gldm','ngtdm']
    
    def getPYRAD(self):
        v=self.getROIvalue()
        im=ima.Imaginable(self.getImage())
        roi=ima.Roiable(self.getROI(),roivalue=v)


#   - distances [[1]]: List of integers. This specifies the distances between the center voxel and the neighbor, for which
#     angles should be generated.
#   - symmetricalGLCM [True]: boolean, indicates whether co-occurrences should be assessed in two directions per angle,
#     which results in a symmetrical matrix, with equal distributions for :math:`i` and :math:`j`. A symmetrical matrix
#     corresponds to the GLCM as defined by Haralick et al.
#   - weightingNorm [None]: string, indicates which norm should be used when applying distance weighting.
#     Enumerated setting, possible values:

#     - 'manhattan': first order norm
#     - 'euclidean': second order norm
#     - 'infinity': infinity norm.
#     - 'no_weighting': GLCMs are weighted by factor 1 and summed
#     - None: Applies no weighting, mean of values calculated on separate matrices is returned.

        if self.Options["min"]=='N':
            self.Options["min"]=im.getMinimumValue()
        if self.Options["max"]=='N':
            self.Options["max"]=im.getMaximumValue()
        settings = {
                    'nbins': self.Options["bin"],
                    'kernelRadius': self.Options["radius"],
                    'distances': [self.Options["radius"]],
                    'min': self.Options["min"],
                    'minimum': self.Options["min"],
                    'max': self.Options["max"],
                    'maximum': self.Options["max"]
                }
        #set min and max pyradiomics


        extractor = prsfe.RadiomicsFeatureExtractor(**settings)
        extractor.enableAllFeatures()
        extractor.enableAllImageTypes()
        # extractor.enableFeatureClassByName(self.featurestype)
        #get extractor imagestypes
        
        # roi.resampleOnTargetImage(im)
        if im.isImaginableInTheSameSpace(roi):
            roi.resampleOnTargetImage(im)
            
        P = extractor.execute(im.getImage(), roi.getImage())
        
        O={}
        for k,v in P.items():
            if not('diagnostic' in k):
                O[k]=float(v)
                    
        return O
                

    def getFeatures(self,moredomain=None):
        MASK=''
        if not moredomain:
            moredomain=''

        K=self.getPYRAD()
        DOMAIN=moredomain + "PYRAD"
        O={DOMAIN:K}
        return O

def geRoiValues(im,roi,v=1):
    """_summary_
    extravt the values in the roi

    Args:
        im (str): image position
        roi (str): region position
                """
    IM=ima.Imaginable(im)
    R=ima.Roiable(roi,roivalue=v)
    if IM.isImaginableInTheSameSpace(R):
        R.resampleOnTargetImage(IM)
    return IM.getValuesInRoi(R)
#pip install benford_py
import benford as bf
def getBenford(v):
    L=bf.Benford((v,'roi'))
    return L.F1D, L.SD
def getChi(found,expected,values_out=False):
    K=np.power((found-expected),2)/expected
    if values_out:
        K.index=[f'Chi_{i}' for i in K.index]
        O=(np.sum(K),K)
    else:
        O=K
    return O

def getBefordStats(v):
    df=pd.DataFrame(v,columns=['roi'])
    bf.Benford((df,'roi'))

class BenfordFE(BD2DecideFE):
    def getFeatures(self,moredomain=None):
        MASK=''
        if not moredomain:
            moredomain=''
        
        im=self.getImage()
        roi=self.getROI()
        v=self.getROIvalue()
        roivalues=geRoiValues(im,roi,v)
        #to dataframe
        roivalues=[float(x) for x in roivalues]
        df=pd.DataFrame(roivalues,columns=['roi'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        (BF,SD)=getBenford(df)

        BF.sort_index(inplace=True)
        Chi,K=getChi(BF.Found,BF.Expected,True)
        DOMAIN=moredomain + "Benford"
        O={DOMAIN:K.to_dict()}
        O[DOMAIN]['Chi_stat']=Chi
        for k,v in SD.AbsDif.to_dict().items():
            O[DOMAIN][f'AbsoluteDiff_{k}']=v
        return O


import multiprocessing
def exrtactMyFeatures(jf,dimension,parallel=True):
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
    # f(L["dataset"][0]) #THEDEBUGAREA
    if parallel:
        with multiprocessing.Pool() as p:
            res = p.map(f,(L["dataset"]))
        p.close()
    else:
        res=[]
        for l in L["dataset"]:
            o=f(l)
            res.append(o)
    result=[]
    idx=[]
       
    for r,id in res:   
        
        #if r is not empty append it to the result
        if r:
            if (isinstance(id,list)):
                for rr,idr in zip(r,id):
                    result.append(rr)
                    idx.append(idr)
            else:
                result.append(r)
                idx.append(id)
    return result,idx

import pandas as pd
def exrtactMyFeaturesToPandas(jf,dimension,max_level=2,parallel=True):
    r,ind=exrtactMyFeatures(jf,dimension,parallel)
    print("normalizing")
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
    # try:
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
                print("----------",im,roi)
                IM=imaginable.SITKImaginable(filename=im)
                ROI=dev.LabelMapable(filename=roi)
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
    #print exception Error


    # except:
    #     #print the image name
    #     print(X["id"])
    #     return r,ids
        
    return r,ids

    
    
def computeRow(line,d):
    out={}
    for x in line:
        if isinstance(x["groups"],list):
            TD=[[s["type"].lower(),s["options"],s["name"]] for s in x["groups"]]
        else:
            TD=[[s["type"].lower(),s["options"],s["name"]] for s in [x["groups"]]]
        for a,o,name in TD:
            if a=="ss":
                L=SS(d)
            if a=="fos":
                L=FOS(d)
            if a=="glcm":
                L=GLCM(d)
            if a=="glrlm":
                L=GLRLM(d)
            if a=="benford":
                L=BenfordFE(d)
            if (a=="pyradiomic") or (a=="pyrad"):
                L=PYRAD()

            L.setImage(x["image"])
            L.setROI(x["labelmap"])

            L.setROIvalue(x["labelmapvalue"])
            L.setOptions(o)
            f=L.getFeatures()
            ft=list(f.keys())[0]
            prefixname=x["groupPrefix"] +"_"+name
            if not (prefixname in out.keys()):
                out[prefixname]={}
            if not (ft in out[prefixname].keys()):
                out[prefixname][ft]={}
            out[prefixname][ft]= f[ft]
        

    return out


if __name__=="__main__":
    # PA=pd.DataFrame()
    # C=pd.DataFrame()
    # import pyable_eros_montin.imaginable as ima

    # # from benfordslaw import benfordslaw
    # a=pn.Pathable('/data/MYDATA/fulldixon-images/')
    # for d in a.getDirectoriesInPath():
    #     im=f'{d}/data/wo.nii'
    #     roi=f'{d}/data/roi.nii.gz'
    #     v=2
    #     b=BenfordFE()
    #     b.setImage(im)
    #     b.setROI(roi)
    #     b.setROIvalue(v)
    #     F=b.getFeatures()
    #     L=pd.DataFrame.from_dict(F,orient='index')
    #     V=pn.Pathable(d+'/')
    #     P=V.getLastPath()
    #     L.index=[P]
    #     if P[0]=='P':
    #         PA=pd.concat((PA,L))
    #     else:
    #         C=pd.concat((C,L))


    # print(C.mean())
    # print(PA.mean())

    # import matplotlib.pyplot as plt
    # for g in PA.columns:
    #     plt.boxplot((C[g],PA[g]))
    #     plt.title(g)
    #     plt.pause(2)



    # # Load the image and mask.
    # im='/data/MYDATA/fulldixon-images/C-1/data/wo.nii'
    # roi='/data/MYDATA/fulldixon-images/C-1/data/roi.nii.gz'
  
    # b=PYRAD(3)
    # v=2
    # b.setImage(im)
    # b.setROI(roi)
    # b.setROIvalue(v)
    # F=b.getFeatures()
    # print(F)

    import utils
    roilist=[]
    ids=[]
    imageslist=[]

    for tkr in ['NonTKR']:
        S=pn.Pathable(f'/data/MYDATA/Eros_143TKR_143NonTKR/2_Label_Maps_Remapped/{tkr}/9003380.nii.gz')
        for l in S.getFilesInPathByExtension()[0:100]:
            L=pn.Pathable(l)
            r=L.getPosition()
            roilist.append([r])
            L.renamePath('2_Label_Maps_Remapped','4_TSE_SAG_data')
            if not L.exists():
                #remove the roi because there's not the associated image
                roilist.pop()
                break
            im=[]
            im.append(L.getPosition())
            ids.append(f'_{tkr}_{L.getFileName()}')
            imageslist.append(im)

    PT='CONF/002'
    EXTRACTION=f'{PT}/extraction.json'
    P= pn.Pathable(f'{PT}/results.csv')

    DIMENSION=3
    method={'rois_roivalues':'cross','images_confs':'cross','images_rois':'cross'}
    A=utils.MakeJsonFe(method=method)
    A.imageslist=imageslist
    A.roislist=roilist
    A.ids=ids
    omo={"min":"N","max":"N","bin":32}
    MO=[
    {"type":"FOS","options":omo,"name":"FOSBD64"},
    {"type":"SS","options":omo,"name":"SSBD"},
    {"type":"PYRAD","options":omo,"name":"PYRAD"},
    {"type":"benford","options":omo,"name":"BENFORD"},
    ]


    A.confslist=MO
    A.roisvalueslist=[10,20,30,40,50,60]
    D=A.getDictionary()
    o=pn.Pathable(f'{PT}/feconf.json')
    o.ensureDirectoryExistence()
    o.writeJson({"dimension":DIMENSION,"dataset":D})
    p=exrtactMyFeaturesToPandas(o.getPosition(),DIMENSION,3,False)
    p.to_json(EXTRACTION)
