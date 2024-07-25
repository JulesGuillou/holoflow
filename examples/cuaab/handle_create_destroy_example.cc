#include "cuaab.hh"
#include <iostream>

int main() {
  // Create a new CUAAB library context.
  dh::cuaabHandle_t handle;
  dh::cuaabStatus_t status = dh::cuaabCreate(&handle);
  if (status != dh::CUAAB_SUCCESS) {
    std::cerr << "Failed to create CUAAB library context: "
              << dh::cuaabGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }

  // Use it here...

  // Destroy the CUAAB library context.
  status = dh::cuaabDestroy(handle);
  if (status != dh::CUAAB_SUCCESS) {
    std::cerr << "Failed to destroy CUAAB library context: "
              << dh::cuaabGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }
}