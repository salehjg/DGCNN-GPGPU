# Multiplatform Implementation Summary

## Task List
- [ ] blah blah blah


## Ops
Op Name | Batch | CPU | CUDA | OCL | Shape1 | Shape2 | Comb | Sett1 | Val1 | Sett2 | Val2 | Val3 |  Notes |
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |  --- |
Concat2        |Yes No|Yes No|Yes No|Yes No|4D|4D|-|Concat2|3| |-|-|--
Conv2d         |Yes No|Yes No|Yes No|Yes No|4D|-|-|overrideDim2|-1| |-|-|--
DivideTiled    |Yes No|Yes No|Yes No|Yes No|4D,2D|1D|-| |-| |-|-|--
Gather         |Yes No|Yes No|Yes No|Yes No|3D|3D|-|indices_axis|1| |-|-|--
MatAddTiled scr|Yes No|Yes No|Yes No|Yes No|1D|-|-| |-| |-|-|--
MatAddTiled    |Yes No|Yes No|Yes No|Yes No|2D,4D|1D|-| |-| |-|-|--
MatAdd         |Yes No|Yes No|Yes No|Yes No|1D,3D|1D,3D|-| |-| |-|-|--
Matmul_Scalar  |Yes No|Yes No|Yes No|Yes No|1D,3D|-|-| |-| |-|-|--
Matmul         |Yes No|Yes No|Yes No|Yes No|2D,3D|2D,3D|-| |-| |-|-|--
MatSubTiled scr|Yes No|Yes No|Yes No|Yes No|1D|-|-| |-| |-|-|--
MatSubTiled    |Yes No|Yes No|Yes No|Yes No|1D,2D,4D|1D|-| |-| |-|-|--
MatSub         |Yes No|Yes No|Yes No|Yes No|1D,4D|1D,4D|-| |-| |-|-|--
Mean           |Yes No|Yes No|Yes No|Yes No|2D,4D|-|{1-0-0-0},{1-1-1-0}| |-| |-|-|--
MultiplyTiled  |Yes No|Yes No|Yes No|Yes No|2D,4D|1D|-| |-| |-|-|--
ReduceMax      |Yes No|Yes No|Yes No|Yes No|4D|-|-|reductionDim|1,2| |-|-|--
ReduceSum4D    |Yes No|Yes No|Yes No|Yes No|4D|-|{1-1-1-0}| |-| |-|-|--
ReduceSum      |Yes No|Yes No|Yes No|Yes No|2D,3D|-|{3D: 0-0-1},{2D: 0-1-0}| |-| |-|-|--
ReLU           |Yes No|Yes No|Yes No|Yes No|2D,4D|-|-| |-| |-|-|--
Square         |Yes No|Yes No|Yes No|Yes No|3D|-|-| |-| |-|-|--
Tile           |Yes No|Yes No|Yes No|Yes No|3D,4D|-|-|tileAxis|1,2|tileCount|20,1024|-|--
TopK           |Yes No|Yes No|Yes No|Yes No|3D|-|-|axis|2|k|20|-|--
Transpose      |Yes No|Yes No|Yes No|Yes No|3D|-|-| |-| |-|-|--
Variance       |Yes No|Yes No|Yes No|Yes No|2D,4D|-|{2D: 1-0-0-0}{4D: 1-1-1-0}| |-| |-|-|--

