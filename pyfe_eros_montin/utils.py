from itertools import permutations
omo={"min":0,"max":32000,"bin":32}
omo2={"min":0,"max":32000,"bin":128}
MO=[{"name":"FOS","options":omo},{"name":"FOS128","options":omo2},{"name":"GLCM","options":omo},{"name":"GLRLM","options":omo},{"name":"GLCM128","options":omo2},{"name":"GLRLM128","options":omo2}]


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
    def getDictionary(self):
        ROIS=[]
        if self.method['rois_roivalues']=='cross':
            ROIS=[]
            for a in self.roislist:
                line=[]
                for roi in a:
                    for value in self.roisvalueslist:
                            line.append([roi,value])
                ROIS.append(line)
        if self.method['images_confs']=='cross':
            IMAGES=[]
            for a in self.imageslist:
                line=[]
                for image in a:
                    for conf in self.confslist:
                            line.append([image,conf])
                IMAGES.append(line)
        OUT=[]
        if self.method['images_rois']=='cross':
            
            for images,rois,id in zip(IMAGES,ROIS,self.ids):
                out={"data":[]}
                for image in images:
                        for roi in rois:
                                out['data'].append({"image":image[0],"labelmapvalue":roi[1],"groups":image[1],"groupPrefix":"","labelmap":roi[0]})
                out["id"]=id
                if self.AUG is not None:
                    out["augment"]=self.AUG      
                OUT.append(out)
                
#         out2["id"]=id
#         out1["augment"]={"n":100,"type":"able","options":{"r":[[-5,5]],"t":[[-5,5],[-5,5]]}}
#         out2["augment"]={"n":100,"type":"able","options":{"r":[[-5,5]],"t":[[-5,5],[-5,5]]}}
        return OUT

import pynico_eros_montin.pynico as pn
import pyfe_eros_montin.pyfe as pf
if __name__=='__main__':
    A=MakeJsonFe()
    D=A.getDictionary()
    o=pn.Pathable('/g/a.json')
    o.writeJson({"dimension":2,"dataset":D})
    p=pf.exrtactMyFeaturesToPandas(o.getPosition(),2,3)
    p.to_json('/g/extr.json')
# from copy import copy
# from pynico_eros_montin import pynico as pn
# import copy
# from pyable_eros_montin import imaginable as able
# opo={"min":-32000,"max":32000,"bin":32}
# opo2={"min":-32000,"max":32000,"bin":128}
# PO=[{"name":"FOS","options":opo},{"name":"FOS128","options":opo2},{"name":"GLCM","options":opo},{"name":"GLRLM","options":opo},{"name":"GLCM128","options":opo2},{"name":"GLRLM128","options":opo2}]
# omo={"min":0,"max":32000,"bin":32}
# omo2={"min":0,"max":32000,"bin":128}
# MO=[{"name":"FOS","options":omo},{"name":"FOS128","options":omo2},{"name":"GLCM","options":omo},{"name":"GLRLM","options":omo},{"name":"GLCM128","options":omo2},{"name":"GLRLM128","options":omo2}]


# def dilateRois(x,r=2):
#     fn=pn.Pathable(x)
#     R=able.Roiable(fn.getPosition())
#     R.dilateRadius(r)
#     fn.addPrefix(f'dilate{r:02d}')
#     R.writeImageAs(fn.getPosition())
#     return fn.getPosition()


# def writeTDCSJson(directory,manjson,bbjson):
#     H=pn.Pathable(directory)
#     O1=[]
#     O2=[]
#     for p in H.getDirectoriesInPath():
#         print(p)
#         L=pn.Pathable(p)
#         L.addBaseName('a.nii')
#         out1={}
#         out2={}
#         id=L.getLastPath()
#         out1["id"]=id
#         out2["id"]=id
#         out1["augment"]={"n":100,"type":"able","options":{"r":[[-5,5]],"t":[[-5,5],[-5,5]]}}
#         out2["augment"]={"n":100,"type":"able","options":{"r":[[-5,5]],"t":[[-5,5],[-5,5]]}}
#         out1["data"]=[]
#         out2["data"]=[]

#     #    rois
#         pimphalca=L.getFilesInPathByExtensionAndPattern('PRE_LCA_PHA')[0]
#         pimphalva=L.getFilesInPathByExtensionAndPattern('PRE_LVA_PHA')[0]
#         pimpharca=L.getFilesInPathByExtensionAndPattern('PRE_RCA_PHA')[0]
#         pimpharva=L.getFilesInPathByExtensionAndPattern('PRE_RVA_PHA')[0]

#         mimphalca=L.getFilesInPathByExtensionAndPattern('PRE_LCA_MAG')[0]
#         mimphalva=L.getFilesInPathByExtensionAndPattern('PRE_LVA_MAG')[0]
#         mimpharca=L.getFilesInPathByExtensionAndPattern('PRE_RCA_MAG')[0]
#         mimpharva=L.getFilesInPathByExtensionAndPattern('PRE_RVA_MAG')[0]

        
#         po1={"image":pimphalca,"labelmapvalue":1,"groups":PO,"groupPrefix":"phase_lca"}
#         po2={"image":pimphalva,"labelmapvalue":1,"groups":PO,"groupPrefix":"phase_lva"}
#         po3={"image":pimpharca,"labelmapvalue":1,"groups":PO,"groupPrefix":"phase_rca"}
#         po4={"image":pimpharva,"labelmapvalue":1,"groups":PO,"groupPrefix":"phase_rva"}


#         mo1={"image":mimphalca,"labelmapvalue":1,"groups":MO,"groupPrefix":"magnitude_lca"}
#         mo2={"image":mimphalva,"labelmapvalue":1,"groups":MO,"groupPrefix":"magnitude_lva"}
#         mo3={"image":mimpharca,"labelmapvalue":1,"groups":MO,"groupPrefix":"magnitude_rca"}
#         mo4={"image":mimpharva,"labelmapvalue":1,"groups":MO,"groupPrefix":"magnitude_rva"}
        
        

#         # bblca=L.getFilesInPathByExtensionAndPattern('BB*LCA')[0]
#         # bblva=L.getFilesInPathByExtensionAndPattern('BB*LVA')[0]
#         # bbrca=L.getFilesInPathByExtensionAndPattern('BB*RCA')[0]
#         # bbrva=L.getFilesInPathByExtensionAndPattern('BB*RVA')[0]

#         R=pn.copy.deepcopy(L)
#         R.changeExtension('nii.gz')
#         manlca=R.getFilesInPathByExtensionAndPattern('LCA')[0]
#         manlva=R.getFilesInPathByExtensionAndPattern('LVA')[0]
#         manrca=R.getFilesInPathByExtensionAndPattern('RCA')[0]
#         manrva=R.getFilesInPathByExtensionAndPattern('RVA')[0]

        


#         po1["labelmap"]=dilateRois(manlca,4)
#         po2["labelmap"]=dilateRois(manlva,4)
#         po3["labelmap"]=dilateRois(manrca,4)
#         po4["labelmap"]=dilateRois(manrva,4)

#         mo1["labelmap"]=dilateRois(manlca,4)
#         mo2["labelmap"]=dilateRois(manlva,4)
#         mo3["labelmap"]=dilateRois(manrca,4)
#         mo4["labelmap"]=dilateRois(manrva,4)

#         out1["data"].append(copy.deepcopy(po1))
#         out1["data"].append(copy.deepcopy(po2))
#         out1["data"].append(copy.deepcopy(po3))
#         out1["data"].append(copy.deepcopy(po4))
#         out1["data"].append(copy.deepcopy(mo1))
#         out1["data"].append(copy.deepcopy(mo2))
#         out1["data"].append(copy.deepcopy(mo3))
#         out1["data"].append(copy.deepcopy(mo4))
        
        
#         po1["labelmap"]=manlca
#         po2["labelmap"]=manlva
#         po3["labelmap"]=manrca
#         po4["labelmap"]=manrva

#         mo1["labelmap"]=manlca
#         mo2["labelmap"]=manlva
#         mo3["labelmap"]=manrca
#         mo4["labelmap"]=manrva

#         out2["data"].append(copy.deepcopy(po1))
#         out2["data"].append(copy.deepcopy(po2))
#         out2["data"].append(copy.deepcopy(po3))
#         out2["data"].append(copy.deepcopy(po4))
#         out2["data"].append(copy.deepcopy(mo1))
#         out2["data"].append(copy.deepcopy(mo2))
#         out2["data"].append(copy.deepcopy(mo3))
#         out2["data"].append(copy.deepcopy(mo4))

        
#         O1.append(copy.deepcopy(out1))
#         O2.append(copy.deepcopy(out2))

#     B=pn.Pathable(manjson)
#     B.ensureDirectoryExistence()
#     B.writeJson({"dimension":2,"dataset":O2})

#     C=pn.Pathable(bbjson)
#     C.ensureDirectoryExistence()
#     C.writeJson({"dimension":2,"dataset":O1})


# import os
    
# H=pn.Pathable('/g/testuno/')


# # import pandas as pd

# # from pyfe_eros_montin import pyfe as pf
# # p=pf.exrtactMyFeaturesToPandas(H1.getPosition(),2,3)
# # p2=pf.exrtactMyFeaturesToPandas(M1.getPosition(),2,3)
# # p=p.append(p2)
# # H1.changeBaseName('dataframeXaug.json')
# # p.to_json(H1.getPosition())



# # h2=pf.exrtactMyFeaturesToPandas(H2.getPosition(),2,3)
# # m2=pf.exrtactMyFeaturesToPandas(M2.getPosition(),2,3)
# # h2=h2.append(m2)
# # H2.changeBaseName('dataframeXaug.json')
# # h2.to_json(H2.getPosition())
