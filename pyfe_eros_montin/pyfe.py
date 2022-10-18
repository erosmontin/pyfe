import ctypes

import os
from tkinter.messagebox import RETRY
from pynico_eros_montin import pynico as pn

class FE():
    def __init__(self,image=None,roi=None,roiv=None,PT=None,options=None) -> None:
        self.info={
            "type":"origin",
            "author":"eros.montin@gmail.com",
            "git":"",
        }

  
        self.PT=PT        
        self.Options=None
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
            for key in self.Options():
                try: 
                    if o[key]:               
                        setattr(self, key, o[key])
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
        self.min="N"
        self.nt=12
        self.max="N"
        self.bin=32
        self.marginalScale=1
        self.outputN= pn.createRandomTemporaryPathableFromFileName('a.txt',self.PT).getPosition()
        self.outputV= pn.createRandomTemporaryPathableFromFileName('a.txt',self.PT).getPosition()
        self.trash.throw(self.outputN)
        self.trash.throw(self.outputV)
        self.maskOut= None
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
        args=(ctypes.c_char_p*7)(b'ss',self.ROI.encode(),str(self.ROIvalue).encode(),aa.encode(),self.outputV.encode(),self.outputN.encode(),str(self.nt).encode())
        out =self.__execute__(os.path.join(basepath,"bld/libfe3dss.so"),args,moredomain + "SS")
        return out

class FOS(BD2DecideFE):
    def getFeatures(self,moredomain=None):
        MASK=''
        if not moredomain:
            moredomain=''
  
        basepath = os.path.dirname(os.path.abspath(__file__))
        args=(ctypes.c_char_p*10)(b'fos',self.Image.encode(),self.ROI.encode(),str(self.ROIvalue).encode(),
        str(self.min).encode(),str(self.max).encode(),
        str(self.bin).encode(),str(self.marginalScale).encode(),
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
        self.radius=1
        if options:
            self.setOptions(options)

class GLCM(TEXTURES):
     def getFeatures(self,moredomain=None):
        MASK=''
        if not moredomain:
            moredomain=''
        aa="N"
        basepath = os.path.dirname(os.path.abspath(__file__))
        args=(ctypes.c_char_p*12)(b'glcm',self.Image.encode(),self.ROI.encode(),str(self.ROIvalue).encode(),
        str(self.min).encode(),str(self.max).encode(),
        str(self.bin).encode(),str(self.radius).encode(),
        self.outputV.encode(),self.outputN.encode(),
        str(self.nt).encode())

        print(f'glcm {self.Image} {self.ROI} {self.ROIvalue} {self.min} {self.max} {self.bin} {self.radius} {self.outputV} {self.outputN}  {self.nt}')
        
        out={}
        if self.dimension==3:
            out =self.__execute__(os.path.join(basepath,"bld/libfe3dglcm.so"),args,moredomain + "GLCM_" + str(self.radius))
        elif self.dimension==2:
            out =self.__execute__(os.path.join(basepath,"bld/libfe2dglcm.so"),args,moredomain + "GLCM_" + str(self.radius))

        return out


class GLRLM(TEXTURES):
     def getFeatures(self,moredomain=None):
        MASK=''
        if not moredomain:
            moredomain=''
        aa="N"
        basepath = os.path.dirname(os.path.abspath(__file__))
        args=(ctypes.c_char_p*11)(b'glrlm',self.Image.encode(),self.ROI.encode(),str(self.ROIvalue).encode(),
        str(self.min).encode(),str(self.max).encode(),
        str(self.bin).encode(),str(self.radius).encode(),
        self.outputV.encode(),self.outputN.encode(),
        str(self.nt).encode())
        
        out={}
        if self.dimension==3:
            out =self.__execute__(os.path.join(basepath,"bld/libfe3dglrlm.so"),args,moredomain + "GLRLM_" + str(self.radius))
        elif self.dimension==2:
            out =self.__execute__(os.path.join(basepath,"bld/libfe2dglrlm.so"),args,moredomain + "GLRLM_" + str(self.radius))

        return out
        

if __name__=="__main__":
    # P=pn.Pathable('/data/PERSONALPROJECTS/myPackages/pyfe/pyfe_eros_montin/data/data.json')
    # o=P.readJson()
    # x=o["dataset"][0][0]



    # L=GLCM(3)
    # L.setImage(x["image"])
    # L.setROI(x['labelmap'])
    # L.setROIvalue(x["labelmapvalue"])
    # L.setOptions(x["settings"])
    # L.getFeatures()
    # print(L.getFeatures())
   o=fetchMyData('/data/PERSONALPROJECTS/myPackages/pyfe/pyfe_eros_montin/data/data.json')

   P=pn.createRandomTemporaryPathableFromFileName('a.json','/data/tttt')
   P.writeJson(o)



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