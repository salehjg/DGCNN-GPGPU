# OpenCL Reminders
* In NDRange, __N__ can be 1,2 or 3
* __work-item__: It is the individual kernel execution instance
* __work-group__: It is a group of __work-items__ form a __work-group__
---
* __global-id__: A unique global ID given to each __work-item__ in the global NDRange
* __local-id__: A unique local ID given to each __work-item__ within a __work-group__
---
* __Gx*Gy__ number of __work-groups__ in __NDRange__
* __Sx*Sy__ number of __work-items__ in a __work-group__
* 

# Useful Method
* clEnqueueWriteBuffer
* clEnqueueWriteBufferRect
---
* clEnqueueReadBuffer
* clEnqueueReadBufferRect
---
* clEnqueueCopyBuffer
* clEnqueueCopyBufferRect

# Useful Macros
* get_work_dim: Returns the number of dimensions associated with the kernel launch.
* get_global_size: Returns the global number of __work-items__ in dimension specified by argument.
* get_global_id: Returns __global-id__ of the kernel __work-item__ for dimension specified by argument.
* get_local_size: Returns the local number of __work-items__ in a __work-group__ for dimension specified by argument
* get_local_id: Returns the __local-id__ of the kernel __work-item__ in a __work-group__ for dimension specified by argument
* get_num_groups: Returns number of __work-groups__ that are executing with this kernel for the dimension specified by argument
* get_group_id: Returns the __group-id__ of the __work-group__ in the dimension specified by argument
* get_global_offset: Returns the offset values specified in the __global_work_offset__ argument during the launch of the kernel using __clEnqueueNDRangeKernel__ function


# Address Space Qualifiers
* __global: share across whole NDRange
* __local: shared within a work-group
* __constant: share across whole NDRange
* __private: ??????