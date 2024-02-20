import toml

# with open('pyproject.toml', 'r') as f:
#     config = toml.load(f)
 
# # Access values from the config
# VERSION=config['project']['version']

from itertools import permutations
omo={"min":0,"max":32000,"bin":32}
omo2={"min":0,"max":32000,"bin":128}
MO=[{"name":"FOS","options":omo},{"name":"FOS128","options":omo2},{"name":"GLCM","options":omo},{"name":"GLRLM","options":omo},{"name":"GLCM128","options":omo2},{"name":"GLRLM128","options":omo2}]

def extractfeatures(imagelist,roilist,conflist,roisvaluelist,AUG,ids)->str:
    #imagelist: list of list of strings =[['a','b'],['c',d]]
    #roilist: list of list of strings =[['a','b'],['c',d]]
    #conflist: list of list of strings =[['a','b'],['c',d]]
    #TODO: check if the lists are the same length
    
    return True
class MakeJsonFe():
    def __init__(self,method={'rois_roivalues':'cross','images_confs':'cross','images_rois':'cross'}) -> None:
    # def __init__(self,imageslist,roislist,roisvalueslist,confslist,method={'rois_roivalues':'dot','images_confs':'dot','images_rois':'dot'}) -> None:
        AVAILABLEMETHODS=['cross','dot']
        self.imageslist=[['/g/testuno/HC_sub03_v1/PRE_LCA_PHAs11719935-0022-00001-000001-01.nii','/g/testuno/HC_sub03_v1/PRE_RCA_PHAs11719935-0025-00001-000001-01.nii'],['/g/testuno/HC_sub06_v1/PRE_LCA_PHAs9599865-0022-00001-000001-01.nii','/g/testuno/HC_sub06_v1/PRE_RCA_PHAs9599865-0025-00001-000001-01.nii']]
        self.roislist=[['/g/testuno/HC_sub03_v1/LCA.nii.gz','/g/testuno/HC_sub03_v1/RCA.nii.gz'],['/g/testuno/HC_sub06_v1/LCA.nii.gz','/g/testuno/HC_sub06_v1/RCA.nii.gz']]
        self.confslist=[MO]
        self.roisvalueslist=[1]
        self.method=method
        self.AUG=None
        self.ids=[f'p{n:02d}' for n in range(len(self.imageslist))]
    def setIDs(self,id):
         self.ids=id
    def getDictionary(self):
        #let's see how is the situation with the rois
        ROIS=[]
        manyrois=False
        if len(self.roisvalueslist)>100:
            manyrois=True
        if self.method['rois_roivalues']=='cross':
            # we are going to mix ROIS x rv = (2,1) (3,) => 6 rois x line
            ROIS=[]
            for pa,a in enumerate(self.roislist):
                line=[]
                for pr,roi in enumerate(a):
                    for pv,value in enumerate(self.roisvalueslist):
                            if manyrois:
                                line.append([roi,value,f'R:{pr:03d}_RV:{pv}'])
                            else:
                                line.append([roi,value,f'R:{pr:02d}_RV:{pv:02d}'])
                            print(line[-1][-1])
                            
                ROIS.append(line)
        #let's see how is the situation with the images and confs
        if self.method['images_confs']=='cross':
            IMAGES=[]
            for pa,a in enumerate(self.imageslist):
                line=[]
                for pi,image in enumerate(a):
                    for pc,conf in enumerate(self.confslist):
                            line.append([image,conf,f'I:{pi:02d}_C:{pc:02d}'])
                IMAGES.append(line)
        OUT=[]
        if self.method['images_rois']=='cross':
            
            for images,rois,id in zip(IMAGES,ROIS,self.ids):
                out={"data":[]}
                for ni,image in enumerate(images):
                        for nr,roi in enumerate(rois):
                                out['data'].append({"image":image[0],"labelmapvalue":roi[1],
                                                    "groups":image[1],"groupPrefix":f"group{image[2]}_{roi[2]}","labelmap":roi[0]})
                out["id"]=id
                if self.AUG is not None:
                    out["augment"]=self.AUG      
                OUT.append(out)
                
        return OUT
    def makeAUG(self,n,r,t,s):
        self.AUG={"n":n,"r":r,"t":t,"s":s}
import pynico_eros_montin.pynico as pn
try:
     #debug
     from pyfe import *
     from pyml import *
except:
     #prod
     from pyfe_eros_montin.pyfe import *
     from pyfe_eros_montin.pyml import *


if __name__=='__main__':
    RESULTSOUTPUT='/g/CONF/005dess/'
    a=pn.Pathable('/g/CONF/005dess/images.json')
    L=a.readJson()
    images=L["images"]
    rois=L["rois"]
    ids=L["ids"]
    rv=[1]

    # create the pyfe object


    EXTRACTION=f'{RESULTSOUTPUT}/extraction.json'


   
    DIMENSION=3
    method={'rois_roivalues':'cross','images_confs':'cross','images_rois':'cross'}
    A=MakeJsonFe(method=method)
    A.imageslist=images
    A.roislist=rois
    A.ids=ids
    A.AUG={"n":2,"options":{
        "r":[[-5,5]]*3,
        "t":[[-5,5]]*3,
        "s":[[0.9,1.1]]*3,
        "noise":{ "type":"gaussian","options":{"mean":0,"std":0.1}},
        "resample":[5,5,5]
           }
    }
    conf={"min":"N","max":"N","bin":16}
    CONF=[
    {"type":"PYRAD","options":conf,"name":"PYRAD"},
    # {"type":"benford","options":omo,"name":"BENFORD"},
    ]
    A.confslist=CONF
    A.roisvalueslist=rv
    D=A.getDictionary()
    o=pn.Pathable(f'{RESULTSOUTPUT}/feconf.json')
    o.ensureDirectoryExistence()
    o.writeJson({"dimension":DIMENSION,"dataset":D})
    db=f'{RESULTSOUTPUT}/db.sqlite'
    l=pn.Log()
    l.fn='/g/a.log' 
    p=exrtactMyFeaturesToSQLlite(o.getPosition(),DIMENSION,3,parallel=False,augonly=False,db='/g/sq.sqlite',table_name='features',log=l)

    # roilist=[]
    # ids=[]
    # imageslist=[]

    # for tkr in ['TKR','NonTKR']:
    #     S=pn.Pathable(f'/data/MYDATA/Eros_143TKR_143NonTKR/2_Label_Maps_Remapped/{tkr}/9003380.nii.gz')
    #     for l in S.getFilesInPathByExtension()[0:2]:
    #         L=pn.Pathable(l)
    #         r=L.getPosition()
    #         roilist.append([r])
    #         L.renamePath('2_Label_Maps_Remapped','1_T2ValuesMaps')
    #         if not L.exists():
    #             #remove the roi because there's not the associated image
    #             roilist.pop()
    #             break
    #         im=[]
    #         im.append(L.getPosition())
    #         ids.append(f'_{tkr}_{L.getFileName()}')
    #         imageslist.append(im)




    # DIMENSION=3
    # method={'rois_roivalues':'cross','images_confs':'cross','images_rois':'cross'}
    # A=MakeJsonFe(method=method)
    # A.imageslist=imageslist
    # A.roislist=roilist
    # A.ids=ids
    # omo={"min":0,"max":5000,"bin":32}
    # omo2={"min":0,"max":5000,"bin":128}
    # MO=[
    # {"type":"FOS","options":omo,"name":"FOS32"},
    #     {"type":"FOS","options":omo2,"name":"FOS128"},
    #     {"type":"GLCM","options":omo,"name":"GLCM32"},
    #     {"type":"GLRLM","options":omo,"name":"GLRLM32"},
    #     {"type":"GLCM","options":omo2,"name":"GLCM32"},
    #     {"type":"GLRLM","options":omo2,"name":"GLRLM32"},
    #     {"type":"SS","options":None,"name":"SS_1"}]
    # A.confslist=[MO]
    # # A.roisvalueslist=[10,20,30,40,50,60]
    # A.roisvalueslist=[10,20]
    # D=A.getDictionary()
    # o=pn.Pathable('/g/feconf.json')
    # o.ensureDirectoryExistence()
    # o.writeJson({"dimension":DIMENSION,"dataset":D})
    # p=exrtactMyFeaturesToPandas(o.getPosition(),DIMENSION,3,False)
    # p.to_json('/g/extraction.json')

    # from glob import glob 
    
    # import os
    # theimages=['fo.nii','IN.nii','OUT.nii','wo.nii']

    # P=['/data/PROJECTS/HIPSEGENTATION/RADIOMIC_HIP_2/FILES/C','/data/PROJECTS/HIPSEGENTATION/RADIOMIC_HIP_2/FILES/P']
    # PNAME=['subjects','patients']
    # roilist=[]
    # ids=[]
    # imageslist=[]

    # for p,n in zip(P,PNAME):
    #     for l in glob(p+"*", recursive = True)[0:2]:
    #         r=os.path.join(l,'data/roi.nii.gz')
    #         roilist.append([r])
    #         imageslist.append([os.path.join(l,'data',im) for im in theimages])
    #         ids.append(f'_{n}_{l.replace(p[:-2],"")}')


    # print('h')
    # EXTRACTION='CONF/extraction.json'
    # P= pn.Pathable('CONF/001/results.csv')
    # DIMENSION=3
    # method={'rois_roivalues':'cross','images_confs':'cross','images_rois':'cross'}
    # A=MakeJsonFe(method=method)
    # A.imageslist=imageslist
    # A.roislist=roilist
    # A.ids=ids
    # omo={"min":0,"max":5000,"bin":32}
    # omo2={"min":0,"max":5000,"bin":128}
    # MO=[
    # {"type":"FOS","options":omo,"name":"FOS32"},
    #     {"type":"FOS","options":omo2,"name":"FOS128"},
    #     {"type":"GLCM","options":omo,"name":"GLCM32"},
    #     {"type":"GLRLM","options":omo,"name":"GLRLM32"},
    #     {"type":"GLCM","options":omo2,"name":"GLCM128"},
    #     {"type":"GLRLM","options":omo2,"name":"GLRLM128"},
    #     {"type":"SS","options":None,"name":"SS_1"}]
    # A.confslist=[MO]
    # A.roisvalueslist=[1,2]
    # D=A.getDictionary()
    # o=pn.Pathable('CONF/feconf.json')
    # o.ensureDirectoryExistence()
    # o.writeJson({"dimension":DIMENSION,"dataset":D})