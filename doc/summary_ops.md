# Multiplatform Implementation Summary

## Task List
- [ ] blah blah blah


## Ops
Op Name        | Batch | CPU  | CUDA  | OCL  |Shape1    | Shape2| Comb                          | Sett1         |Val1   |Sett2      |Val2   | Notes|
---            | ---   | ---  | ---   | ---  | ---      | ---   | ---                           | ---           | ---   | ---       | ---   |  --- |
Concat2        |     No|Yes   |Yes    |    No|4D        |4D     |-                              |Concat2        |3      |           |-      |--
Conv2d         |Yes    |Yes   |     No|    No|4D        |-      |-                              |overrideDim2   |-1     |           |-      |--
DivideTiled    |Yes    |Yes   |     No|    No|4D,2D     |1D     |-                              |               |-      |           |-      |--
Gather         |Yes    |Yes   |     No|    No|3D        |3D     |-                              |indices_axis   |1      |           |-      |--
MatAddTiled scr|     No|Yes   |     No|    No|1D        |-      |-                              |               |-      |           |-      |--
MatAddTiled    |     No|Yes   |     No|    No|2D,4D     |1D     |-                              |               |-      |           |-      |--
MatAdd         |     No|Yes   |     No|    No|1D,3D     |1D,3D  |-                              |               |-      |           |-      |--
Matmul_Scalar  |Yes    |Yes   |     No|    No|1D,3D     |-      |-                              |               |-      |           |-      |--
Matmul         |Yes    |Yes   |     No|    No|2D,3D     |2D,3D  |-                              |               |-      |           |-      |--
MatSubTiled scr|     No|Yes   |     No|    No|1D        |-      |-                              |               |-      |           |-      |--
MatSubTiled    |     No|Yes   |     No|    No|1D,2D,4D  |1D     |-                              |               |-      |           |-      |--
MatSub         |     No|Yes   |     No|    No|1D,4D     |1D,4D  |-                              |               |-      |           |-      |--
Mean           |     No|Yes   |Yes    |    No|2D,4D     |-      |{1-0-0-0}, {1-1-1-0}           |               |-      |           |-      |--
MultiplyTiled  |Yes    |Yes   |     No|    No|2D,4D     |1D     |-                              |               |-      |           |-      |--
ReduceMax      |     No|Yes   |Yes    |    No|4D        |-      |-                              |reductionDim   |1,2    |           |-      |--
ReduceSum4D    |     No|Yes   |Yes    |    No|4D        |-      |{1-1-1-0}                      |               |-      |           |-      |--
ReduceSum      |     No|Yes   |Yes    |    No|2D,3D     |-      |{3D: 0-0-1}, {2D: 0-1-0}       |               |-      |           |-      |--
ReLU           |Yes    |Yes   |     No|    No|2D,4D     |-      |-                              |               |-      |           |-      |--
Square         |Yes    |Yes   |     No|    No|3D        |-      |-                              |               |-      |           |-      |--
Tile           |     No|Yes   |  50%  |    No|3D,4D     |-      |-                              |tileAxis       |1,2    |tileCount  |20,1024|only tileAxis=2 implemented
TopK           |Yes    |Yes   |     No|    No|3D        |-      |-                              |axis           |2      |k          |20     |--
Transpose      |Yes    |Yes   |Yes    |    No|3D        |-      |-                              |               |-      |           |-      |--
Variance       |     No|Yes   |Yes    |    No|2D,4D     |-      |{2D: 1-0-0-0}, {4D: 1-1-1-0}   |               |-      |           |-      |--

