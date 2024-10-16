from copy import deepcopy
import ctypes
import json

import os
import numpy
from pynico_eros_montin import pynico as pn

from pydaug_eros_montin import pydaug as pda

import pyable_eros_montin.dev as dev
import toml

import uuid


def getTIME():
    return uuid.uuid4().hex
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
import warnings
import radiomics
class PYRAD(TEXTURES):
    def __init__(self, image=None, roi=None, roiv=None, PT=None, options=None) -> None:
        super().__init__(image, roi, roiv, PT, options)
        self.featurestype=['firstorder','shape','glcm','glrlm','glszm','gldm','ngtdm']
        self.Options["normalize"]=False

    def getPYRAD(self):
        
        v=self.getROIvalue()
        im=ima.Imaginable(self.getImage())
        roi=ima.Roiable(self.getROI(),roivalue=v)
        
        warnings.simplefilter('ignore', DeprecationWarning)
        # logger = radiomics.logging.getLogger("radiomics")
        # logger.setLevel(radiomics.logging.ERROR)
        radiomics.setVerbosity(60)


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

        
            
        
        if "normalize" in self.Options.keys():
            pass
        else:
            self.Options["normalize"]=False
            
            
            
        settings = {
                    'nbins': self.Options["bin"],
                    'binCount': self.Options["bin"],
                    'kernelRadius': self.Options["radius"],
                    'distances': [self.Options["radius"]],
                    'normalize': self.Options["normalize"]
                        }


            
            
        
        settings["interpolator"]=sitk.sitkNearestNeighbor
        
        
        extractor = prsfe.RadiomicsFeatureExtractor(**settings)
        extractor.enableAllFeatures()
        extractor.enableAllImageTypes()
        # extractor.enableFeatureClassByName(self.featurestype)
        #get extractor imagestypes
        
        try:
            if not im.isImaginableInTheSameSpace(roi):
                roi.resampleOnTargetImage(im)

            if roi.getNumberOfNonZeroVoxels()>20:
                # roi.resampleOnTargetImage(im)
                P = extractor.execute(im.getImage(), roi.getImage())
                O={}
                for k,v in P.items():
                    if not('diagnostic' in k):
                        O[k]=float(v)
            else:
                p=roi.getImageAsNumpy()
                if roi.getImageDimension()==3:
                    p[1:5,1:5,1:5]=1
                elif roi.getImageDimension()==2:
                    p[1:5,1:5]=1
                else:
                    p[1:5]=1
                roi.setImageFromNumpy(p)
                P = extractor.execute(im.getImage(), roi.getImage())
                O={}
                for k,v in P.items():
                    if not('diagnostic' in k):
                        O[k]=np.nan
        except:
            p=roi.getImageAsNumpy()
            if roi.getImageDimension()==3:
                p[1:5,1:5,1:5]=1
            elif roi.getImageDimension()==2:
                p[1:5,1:5]=1
            else:
                p[1:5]=1
            roi.setImageFromNumpy(p)
            P = extractor.execute(im.getImage(), roi.getImage())
            O={}
            for k,v in P.items():
                if not('diagnostic' in k):
                    O[k]=np.nan
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
def exrtactMyFeatures(jf,dimension,parallel=True,augonly=False,saveimages=None):
    if isinstance(jf,str):
        P=pn.Pathable(jf)
        if not P.exists():
            raise Exception("file do not exists")
        L=P.readJson()
    elif (isinstance(jf,dict)):
        L=jf
    # if augonly:
    #     if dimension==3:
    #         f=theF3AUG
    #     if dimension==2:
    #         f=theF2AUG
    # else:    
    #     if dimension==3:
    #         f=theF3
    #     if dimension==2:
    #         f=theF2
    # f(L["dataset"][0]) #THEDEBUGAREA
    if parallel:
        from itertools import repeat
        with multiprocessing.Pool() as p:
            n=len(L["dataset"])
            res = p.starmap(theF,zip(L["dataset"],[dimension]*n,[augonly]*n,[saveimages]*n))
        p.close()
    else:
        res=[]
        for l in L["dataset"]:
            o=theF(l,dimension,augonly,saveimages)
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
def exrtactMyFeaturesToPandas(jf,dimension,max_level=3,parallel=True,augonly=False,saveimages=None):
    if saveimages!=None:
        LL=pn.Pathable(saveimages)
        LL.ensureDirectoryExistence()
    r,ind=exrtactMyFeatures(jf,dimension,parallel,augonly=augonly,saveimages=saveimages)
    print("normalizing")
    X=pd.json_normalize(r,max_level=max_level)
    X.index=ind
    return X


import sqlite3
def check_database(db=None):
    if db is None:
        db = 'default.db'
    conn = sqlite3.connect(db)
    return conn,{"db":db}

def check_table(db=None, table_name='extraction'):
    
    conn,conf = check_database(db=db)
    cursor = conn.cursor()

    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    result = cursor.fetchone()

    if result:
        print(f"Table '{table_name}' exists.")
        #get the cursor
        
    else:
        print(f"Table '{table_name}' does not exist. Creating it now.")
        cursor.execute(f"""
            CREATE TABLE {table_name} (
                extraction_id string,
                json_structure TEXT,
                update_date TEXT
            );
        """)
        conn.commit()
        print(f"Table '{table_name}' created.")

    
    conf['table_name'] = table_name
    return conn,conf

def insert_into_table(db, table_name, extraction_id, json_structure):
    conn,conf = check_table(db=db, table_name=table_name)
    cursor = conn.cursor()

    cursor.execute(f"""
        INSERT INTO {table_name} (extraction_id, json_structure, update_date)
        VALUES (?, ?, CURRENT_TIMESTAMP);
    """, (extraction_id, json_structure))
    conn.commit()
    conn.close()
    return conf


def get_all_ids_in_task(jf):
    return [d["id"] for d in jf["dataset"]]
def filter_dataset_batches(jf,db,table_name):
    conn,conf = check_table(db=db,table_name=table_name)
    cursor = conn.cursor()
    ids = []
    for _id in get_all_ids_in_task(jf):
        cursor.execute(f"SELECT extraction_id FROM {table_name} where extraction_id == ?",(_id,))
        for r in cursor.fetchall():
            ids.append(_id)
    conn.close()
    dataset = [d for d in jf["dataset"] if d["id"] not in ids]
    jf["dataset"] = dataset
    return jf
import multiprocessing
def dataset_to_datasetbatches(JF,dimension, parallel):    
    dataset = JF['dataset']
    if not parallel:
        num_cores = 1
    else:
        num_cores = multiprocessing.cpu_count()
    new_dataset = []
    for i in range(0, len(dataset), num_cores):
        batch = dataset[i:i+num_cores]
        # process batch
        j={"dimension":dimension,"dataset":batch}
        new_dataset.append(j)
    return new_dataset        
def exrtactMyFeaturesToSQLlite(jf,dimension,max_level=3,parallel=True,augonly=False,saveimages=None,db=None,table_name='extraction',extraction_configurations=None,log=None):
    # create a database in memory in case user doesn't pass it as an argument
    LOG= log is not None
    if LOG:
        if isinstance(log,str):
            f=log
            log=pn.Log()
            log.fn=f
            
            
    conn,conf=check_table(db=db, table_name=table_name)
    db=conf['db']
    table_name=conf['table_name']
    if saveimages!=None:
        LL=pn.Pathable(saveimages)
        LL.ensureDirectoryExistence()
    #read the json file and extract the features
    JF=pn.Pathable(jf)
    JF=JF.readJson()
    
    


    # get the number of cores available
    B=dataset_to_datasetbatches(JF, dimension,parallel)
    for b in B:
        b=filter_dataset_batches(b,db,table_name)
        if b["dataset"]==[]:
            print("continue")
            continue
        else:
            rs,inds=exrtactMyFeatures(b,dimension,parallel,augonly=augonly,saveimages=saveimages)
            if LOG:
                log.append(f"Extracted {len(inds)} features\n")
                log.dump()
            for r,ind in zip(rs,inds):
                #insert into sqlite
                insert_into_table(db=db, table_name=table_name, extraction_id=ind, json_structure=json.dumps(r))
    return   conf


def pyfejsonFeaturesToPandas(xf,idf,max_level=3):
    P=pn.Pathable(xf)
    I=pn.Pathable(idf)
    if P.exists():
        X=pd.json_normalize(P.readJson(),max_level=max_level)
        X.index=pd.json_normalize(I.readJson(),max_level=max_level)
    return X


from pyable_eros_montin import imaginable
import numpy as np
import SimpleITK as sitk
def theF(X,d,augonly=False,saveimages=None):
    # try:
    line=X["data"]
    r=[]
    ids=[]

    G=pn.GarbageCollector()
    resample=[1,1,1]
    resampleflag=False
    
    try:
        aug=X["augment"]
    except:
        aug=[]
    if "resample" in aug["options"].keys():
        resampleflag=True
        resample=aug["options"]["resample"]
    line2=deepcopy(line)
    if not augonly:
        #extract the features
        
        
        if resampleflag:
            for i,x in enumerate(line2):
                im,roi=x["image"],x["labelmap"]
                IM=imaginable.Imaginable(filename=im)
                ROI=imaginable.Imaginable(filename=roi)
                ROI.dfltInterpolator=sitk.sitkNearestNeighbor
                ROI.dfltuseNearestNeighborExtrapolator=True
                IM.changeImageSpacing(resample)
                ROI.changeImageSpacing(resample)
                imn=pn.Pathable(im)
                roin=pn.Pathable(roi)
                imn.changePathToOSTemporary().changeFileNameRandom()
                roin.changePathToOSTemporary().changeFileNameRandom()
                G.throw(imn.getPosition())
                G.throw(roin.getPosition())
                IM.writeImageAs(imn.getPosition())
                ROI.writeImageAs(roin.getPosition())
                line2[i]["image"]=imn.getPosition()
                line2[i]["labelmap"]=roin.getPosition()
        
        
        r.append(computeRow(line2,d))
        ids.append(X["id"])

    if "augment" in X.keys():
        aug=X["augment"]
        scale=False
        
        if "r" in aug["options"].keys():
            Rv=aug["options"]["r"]
        else:
            Rv=[[0,0]]*d
        if "t" in aug["options"].keys():
            Tv=aug["options"]["t"]
        else:
            Tv=[[0,0]]*d
        
        Sv=[[1,1]]*d
        if "s" in aug["options"].keys():
            Sv=aug["options"]["s"]
            scale =True
        

            
            
        for n in range(aug["n"]):    
            R=[ np.random.uniform(low=l, high=h, size=1)[0] for l,h in Rv]
            T=[ np.random.uniform(low=l, high=h, size=1)[0] for l,h in Tv]
            S=[1]*3
            if scale:
                S=[ np.random.uniform(low=l, high=h, size=1)[0] for l,h in Sv]
            line2 =deepcopy(line)
            for i,x in enumerate(line2):
                im,roi=x["image"],x["labelmap"]
                IM=imaginable.Imaginable(filename=im)
                ROI=imaginable.Imaginable(filename=roi)
                ROI.dfltInterpolator=sitk.sitkNearestNeighbor
                ROI.dfltuseNearestNeighborExtrapolator=True
                AFF=ima.create_affine_matrix(rotation=R,scaling=S)
                
                IM.transformImageAffine(AFF.flatten().tolist(),translation=T)
                ROI.transformImageAffine(AFF.flatten().tolist(),translation=T)
                if resampleflag:
                    IM.changeImageSpacing(resample)
                    ROI.changeImageSpacing(resample)
                
                imn=pn.Pathable(im)
                roin=pn.Pathable(roi)
                if saveimages==None:
                    imn.changePathToOSTemporary().changeFileNameRandom()
                    roin.changePathToOSTemporary().changeFileNameRandom()
                    G.throw(imn.getPosition())
                    G.throw(roin.getPosition())
                else:
                    imn.changePath(saveimages)
                    roin.changePath(saveimages)
                    imn.addSuffix(f'-aug{n:04}')                    
                    roin.addSuffix(f'-aug{n:04}')                    
                    jj=imn.fork()
                    jj.changeExtension('json')
                    jj.writeJson({
                        "rotation":R,
                        "translation":T,
                        "scaling":S,
                        "resample":resample,
                        "affine":AFF.flatten().tolist()
                    })
                    

                    
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
    G.trash()
    return r,ids

    
    
def computeRow(line,d):
    out={}
    for x in line:
        if isinstance(x["groups"],list):
            TD=[[s["type"].lower(),s["options"],s["name"]] for s in x["groups"]]
        else:
            TD=[[s["type"].lower(),s["options"],s["name"]] for s in [x["groups"]]]
        for a,o,name in TD:
            try:
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
            except:
                f=f'errors_extractionproblems{getTIME()}.txt'
                with open(f,'a') as fi:
                    fi.write(f'{x["image"]},{x["labelmap"]},{x["labelmapvalue"]}\n')
                fi.close()
                continue


    return out


if __name__=="__main__":
    # J='/data/PROJECTS/oai_research/hpc/feconf_000.json'
    J='/g/feconf.json'
    l=pn.Log()
    exrtactMyFeaturesToSQLlite(J,3,3,parallel=False,augonly=False,saveimages=None,db='/g/db.sqlite',table_name='features',log=l)


