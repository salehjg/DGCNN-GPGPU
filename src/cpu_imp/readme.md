# Pure C++ Implementation
This directory contains codes related to __PURE C++ IMPLEMENTATION__, __LINEAR ALGEBRA C++__.


* __model01.cpp:__ 
low level implementation of DGCNN, using pure C++, without any linear algebra library 
and optimization for speed and memory footprint. _class implementation could be found at ./inc/cpu_imp/model01.hpp_

* __model0x.cpp:__
bla blah blah blah!

# Model01 Details
## Task List
- [x] Managing dynamic memory
- [x] Implement adjustable batch offset and batch size
- [ ] Implement new type for tensor with shape details
- [ ] Trying more optimized approach for computing variance
- [ ] Reuse median and/or variance within another methods when possible(like median within variance)
- [ ] Manage file names for matrix dumps
- [ ] Dynamic job assignment on different threads(using dedicated library)

## Methods
Method Name | Description | is a BatchOP | Implemented | Notes
----------- | ----------- | ------------ | ------------ | -----
Pairwise_Distance |  | Yes | Yes | -
Get_Edge_Features |  | Yes | Yes | -
KNN |  | Yes | Yes | -
Conv2D |  | Yes | Yes | -
Batchnorm_Forward |  | No | Yes | -
FullyConnected_Forward |  | Yes | Yes | -
ReLU |  | Yes | Yes | -
TransformNet |  | Yes | Yes | -
LA_transpose |  | Yes | Yes | -
LA_MatMul |  | Yes | Yes | -
LA_Square |  | Yes | Yes | -
LA_Sum |  | No | Yes | -
LA_Sum4D |  | No | Yes | -
LA_Mean |  | No | Yes | -
LA_Variance |  | No | Yes | -
LA_ADD |  | No | Yes | -
LA_SUB |  | No | Yes | -
LA_Concat2 |  | No | Yes | -
LA_ReduceMax |  | No | Yes | -
LA_Tile | | No | No | -


 

