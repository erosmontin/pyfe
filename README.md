# pyFe
Features extraction and machine learning 

## installation
1. install this package
1. compile [Features Extractor](https://github.com/erosmontin/FeaturesExtractor)
1. copy bld directory in the library path
1. install [pyfe](https://www.githu.com/erosmontin/pyfe) or ask [Eros](eros.montin@gmail.com)
```
pip install git+https://www.github.com/erosmontin/pyfe
```


## Changhelog:
1. version 0.1.3 **2023-10-13**
    - pyradiomic PYRAD
1. version 0.1.2 **2023-10-13**
    - Benford Law FE {"type":"BENFORD","options":{"min":0,"max":5000,"bin":128},"name":"BENFORD"}
1. version 0.1.0 **2023-09-20**
    - add the group name in the group = {"type":"FOS","options":{"min":0,"max":5000,"bin":128},**"name":"FOS32"**}
    - change the "name" field in the groups to type and that can be FOS,GLCM,GLRLM or SS {**"type":"FOS"**,"options":{"min":0,"max":5000,"bin":128},"name":"FOS32"}
    - multiprocess of pyml and geenral debug of the classification accuracy function
    - added utils.py, where there's a new class to customize the extraction:
1. vesion 0.0.0 **2022-08-01**
    - main class for exrtaction and machine learning for classification



[*Dr. Eros Montin, PhD*](http://me.biodimensional.com)
**46&2 just ahead of me!**

