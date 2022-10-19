
from pyfe_eros_montin import pyfe as pf
from pynico_eros_montin import pynico as pn


 







#F=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/data.json')
#P=pn.Pathable(F)
#o=P.readJson()

#theF3(o["dataset"][0])
#o=fetchMyData(F)
#print(len(o))
P=pn.Pathable('/data/tttt/a.json')
#P.writeJson(o)


# def iterdict(d,o=None):
#   l=[]
#   for k,v in d.items():        
#      if isinstance(v, dict):
#          l.append(iterdict(v,k))
#      else:            
#          return(o + k,":",v)


#L=pd.json_normalize(P.readJson(),max_level=2)

# [{"name":"SS","options":{}},{"name":"FOS","options":{}},{"name":"GLCM","options":{}},{"name":"GLRLM","options":{}}]


#f=L.to_numpy()