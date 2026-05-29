from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeRandomTransform
import torch 
from mantra.representations.one_skeleton import OneSkeleton 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torchvision.transforms import Compose 

NAME_TO_CLASS_2M = {
        '#^3 T^2': 0, 
        'S^2': 1, 
        '#^4 RP^2': 2, 
        '#^15 RP^2': 3, 
        '#^2 T^2': 4, 
        '#^10 RP^2': 5, 
        '#^5 T^2': 6, 
        '#^6 T^2': 7, 
        'Klein bottle': 8, 
        'T^2': 9, 
        '#^8 RP^2': 10, 
        '#^12 RP^2': 11, 
        '#^7 RP^2': 12, 
        '#^16 RP^2': 13, 
        '#^8 T^2': 14, 
        '#^17 RP^2': 15, 
        'RP^2': 16, 
        '#^5 RP^2': 17, 
        '#^4 T^2': 18, 
        '#^6 RP^2': 19, 
        '#^3 RP^2': 20}


class ManifoldToFace:
    def __call__(self,data):

        data.face = torch.tensor(data.triangulation).T - 1  
        return data

class NameToClass2MTransform:
    """
    Encode the homemorphism type (`name`) as a nominal target for 2-manifolds.
    """

    def __init__(self):
        self.class_dict = NAME_TO_CLASS_2M

    def __call__(self, data: Data):
        assert "name" in data
        data.y = torch.tensor(self.class_dict[data.name])
        return data


ds = ManifoldTriangulations(
    root="./data",      # root folder for storing data
    dimension=2,        # Whether to load 2- or 3-manifolds
    version="latest",    # Which version of the dataset to load
    transform=Compose([OneSkeleton(),ManifoldToFace(),NodeRandomTransform(dim=8),NameToClass2MTransform()])
)



#|%%--%%| <HelGsI1P3n|IZ30nvGcEf>


#|%%--%%| <IZ30nvGcEf|mkkkRqUTVu>

dl = DataLoader(ds,batch_size=10)
for batch in dl:
    break

batch.y

#|%%--%%| <mkkkRqUTVu|umAtI4a15a>
data = ds[0]
print(data)
print(data.triangulation)
print(data.face)
print(data.edge_index)
print(data.x)
print(data.y)

#|%%--%%| <umAtI4a15a|1cfydFk7Sb>

n = []
for data in ds:
    n.append(data.name)





#|%%--%%| <1cfydFk7Sb|od9c4ia9GH>
d = {}
for i,name in enumerate(set(n)):
    d[name] = i 

print(d)
