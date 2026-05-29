# Bugs 

- This results in a wrong triangulation. Though is looks correct. 

'''{python}
from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeRandomTransform
from mantra.transforms.structural_transforms import TriangulationToFaceTransform
from mantra.representations.one_skeleton import OneSkeleton 
from torchvision.transforms import Compose 

class ReduceByOne:
    def __call__(self,data):
        data.face = data.face- 1  

dataset = ManifoldTriangulations(
    root="./data",      # root folder for storing data
    dimension=2,        # Whether to load 2- or 3-manifolds
    version="latest",    # Which version of the dataset to load
    transform=Compose([TriangulationToFaceTransform(),OneSkeleton(),NodeRandomTransform(dim=8)]),
    force_reload=False,
)

data = dataset[0]
print(data.triangulation)
print(data.face)
'''

- NameToClass2MTransform transform not correctly implemented. Does, obviously, not run. 

```
class NameToClass2MTransform:
    """
    Encode the homemorphism type (`name`) as a nominal target for 2-manifolds.
    """

    def __init__(self):
        self.class_dict = NAME_TO_CLASS_2M

    def forward(self, data: Data):
        assert "name" in data
        data.y = torch.tensor(self.class_dict[data.name])
        return data
```


- Error running the dataset with above transform. 
```
❯ uv run src/datasets/mantra2d.py
Traceback (most recent call last):
  File "/home/ernst/projects/nerve-pool/src/datasets/mantra2d.py", line 166, in <module>
    get_dataset(config=config, force_reload=True)
  File "/home/ernst/projects/nerve-pool/src/datasets/mantra2d.py", line 104, in get_dataset
    "num_classes": ds.num_classes,
  File "/home/ernst/projects/nerve-pool/.venv/lib/python3.10/site-packages/torch_geometric/data/in_memory_dataset.py", line 92, in num_classes
    return super().num_classes
  File "/home/ernst/projects/nerve-pool/.venv/lib/python3.10/site-packages/torch_geometric/data/dataset.py", line 184, in num_classes
    data_list = _get_flattened_data_list([data for data in self])
  File "/home/ernst/projects/nerve-pool/.venv/lib/python3.10/site-packages/torch_geometric/data/dataset.py", line 184, in <listcomp>
    data_list = _get_flattened_data_list([data for data in self])
  File "/home/ernst/projects/nerve-pool/.venv/lib/python3.10/site-packages/torch_geometric/data/dataset.py", line 300, in __iter__
    yield self[i]
  File "/home/ernst/projects/nerve-pool/.venv/lib/python3.10/site-packages/torch_geometric/data/dataset.py", line 292, in __getitem__
    data = data if self.transform is None else self.transform(data)
  File "/home/ernst/projects/nerve-pool/.venv/lib/python3.10/site-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
  File "/home/ernst/projects/nerve-pool/src/datasets/mantra2d.py", line 48, in __call__
    data.y = torch.tensor(self.class_dict[data.name])
KeyError: '#^7 RP^2'
```
 
Dictionary contains more elements that the dict provided in the package. 

```
{'#^10 RP^2',
 '#^12 RP^2',
 '#^15 RP^2',
 '#^16 RP^2',
 '#^17 RP^2',
 '#^2 T^2',
 '#^3 RP^2',
 '#^3 T^2',
 '#^4 RP^2',
 '#^4 T^2',
 '#^5 RP^2',
 '#^5 T^2',
 '#^6 RP^2',
 '#^6 T^2',
 '#^7 RP^2',
 '#^8 RP^2',
 '#^8 T^2',
 'Klein bottle',
 'RP^2',
 'S^2',
 'T^2'}
```
vs in the package

```
NAME_TO_CLASS_2M = {
    "Klein bottle": 0,
    "RP^2": 1,
    "T^2": 2,
    "S^2": 3,
    "": 4,
    "#^2 RP^2": 5,
    "#^3 RP^2": 6,
    "#^4 RP^2": 7,
    "#^5 RP^2": 8,
}
```

- Dataset can not concatenate the result of the above transform. 

```
❯ uv run src/datasets/mantra2d.py
Traceback (most recent call last):
  File "/home/ernst/projects/nerve-pool/src/datasets/mantra2d.py", line 176, in <module>
    get_dataset(config=config, force_reload=True)
  File "/home/ernst/projects/nerve-pool/src/datasets/mantra2d.py", line 114, in get_dataset
    "num_classes": ds.num_classes,
  File "/home/ernst/projects/nerve-pool/.venv/lib/python3.10/site-packages/torch_geometric/data/in_memory_dataset.py", line 92, in num_classes
    return super().num_classes
  File "/home/ernst/projects/nerve-pool/.venv/lib/python3.10/site-packages/torch_geometric/data/dataset.py", line 186, in num_classes
    y = torch.cat([data.y for data in data_list if 'y' in data], dim=0)
RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated
```






