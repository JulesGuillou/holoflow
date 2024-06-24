Introduction
============

Welcome to the Holoflow project documentation. This project aims to provide an 
abstraction layer for signal processing model. It provides the user a way to
describe the signal processing model as a tree where nodes are signal processing
functions (computations) and edges are data dependencies. The user can then
execute the model without worrying about how to implement the data flow.

The project is divided in several libraries:

- `holoflow`: The core library that provides the abstraction layer for signal
  processing models.

- `batched_spsc_queue`: A lock-free, single-producer, single-consumer queue
  that supports batched operations.
