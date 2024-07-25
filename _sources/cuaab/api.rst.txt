cuAAB API Documentation
=======================

.. doxygenenum:: dh::cuaabStatus_t
   :project: Holoflow

.. doxygentypedef:: dh::cuaabHandle_t
   :project: Holoflow

.. doxygenfunction:: dh::cuaabCreate(cuaabHandle_t* handle)
   :project: Holoflow

.. doxygenfunction:: dh::cuaabDestroy(cuaabHandle_t handle)
   :project: Holoflow

.. doxygenfunction:: dh::cuaabSetStream(cuaabHandle_t handle, cudaStream_t stream)
   :project: Holoflow

.. doxygenfunction:: dh::cuaabAAB(cuaabHandle_t handle, const float *input, float *output, int stride, int batch)
   :project: Holoflow
