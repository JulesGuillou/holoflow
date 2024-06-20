# Top level TODOs
### Build system (To Complete)

### CI/CD
- [ ] Add a CI/CD pipeline with github actions.
- [ ] Add self-hosted runners for CI/CD pipeline on windows / linux.
- [ ] Explore the use of supervisor to reduce hardware costs.

### Camera / frame grabbers support
- [ ] Add a `phantom_s10_euresys` library to support phantom S710 camera with euresys EGrabber.

### CUDA physics / optics implementation
- [ ] Add a `cufrsld` library that performs Fresnel-Diffraction with cuda-like api.
- [ ] Add a `custft` library that performs Short-Time-Fourrier-Transform with cuda-like api.
- [ ] Add a `cusap` library that performs Spectral-Angular-Propagation with cuda-like api.
- [ ] Add a `cupca` library that performs Principal-Component-Analysis with cuda-like api.

### CUDA wrappers
- [ ] Add a `cudart_utils` library to handle cuda runtime API calls.
- [ ] Add a `cufft_utils` library to handle cufft API calls.
- [ ] Add a `cufrsld_utils` library to handle cufrsld API calls.
- [ ] Add a `custft_utils` library to handle custft API calls.

### Other libs
- [x] Add a `batched_spsc_queue` library to provide a batched single producer single consumer queue for inter-thread tensor communication / tensor resize.

- [ ] Add a `holoflow` library to provide a high-level interface for creating tree-like image processing pipelines where nodes are computationals tasks and edges are data dependencies.

### Apps
- [ ] Add a `holovibes` app that provides a GUI to create / modify image processing pipeline live while fetching data from high-speed cameras of files, with a limited set of options for user convenience.

- [ ] Add a `holodev` app that provides a GUI to create / modify image processing pipeline live with dataflow representation (tree) and a detailed view of each node's parameters. It should expose all the different nodes available in the `holoflow` library.

### Others
- [ ] Add a proper README.md file.
- [ ] Add a LICENSE file.
- [ ] Add a AUTHORS file.
- [ ] Add a CONTRIBUTE.md file.
- [x] Add a TODO.md file.
- [ ] Add a CHANGELOG.md file

# libs/batched_spsc_queue TODOS
- [ ] Generate doxygen documentation via bazel.
- [ ] Add sphinx generate for detailed explanation of library.
- [ ] Add examples for usage.
- [ ] Add a proper README.md file.

# libs/cufrsld TODOS (To Complete)
- [ ] Add an error type to handle errors in the library (similar API than cufft).
- [ ] Add a way to perform Fresnel Diffraction (similar API than cufft).

*See with project physician for a more detailed requirements list.*

# libs/custft TODOS (To Complete)
- [ ] Add an error type to handle errors in the library (similar API than cufft).
- [ ] Add a way to perform Short-Time-Fourrier-Transform (similar API than cufft).

*See with project physician for a more detailed requirements list.*

# libs/cusap TODOS (To Complete)
- [ ] Add an error type to handle errors in the library (similar API than cufft).
- [ ] Add a way to perform Spectral-Angular-Propagation (similar API than cufft).

*See with project physician for a more detailed requirements list.*

# libs/cupca TODOS (To Complete)
- [ ] Add an error type to handle errors in the library (similar API than cufft).
- [ ] Add a way to perform Principal-Component-Analysis (similar API than cufft).

*See with project physician for a more detailed requirements list.*

# libs/cudart_utils TODOS (To Complete)
- [ ] Add a way to handle cuda runtime API calls.
- [ ] Add smart pointers for cuda runtime memory management.
- [ ] Add a way to handle cuda runtime errors with tl:expected.

# libs/cufft_utils TODOS (To Complete)
- [ ] Add a way to handle cufft API calls.
- [ ] Add smart pointers for cufft memory management.
- [ ] Add a way to handle cufft errors with tl:expected.

# libs/cufrsld_utils TODOS (To Complete)
- [ ] Add a way to handle cufrsld API calls.
- [ ] Add smart pointers for cufrsld memory management.
- [ ] Add a way to handle cufrsld errors with tl:expected.

# libs/custft_utils TODOS (To Complete)
- [ ] Add a way to handle custft API calls.
- [ ] Add smart pointers for custft memory management.
- [ ] Add a way to handle custft errors with tl:expected.

# libs/cusap_utils TODOS (To Complete)
- [ ] Add a way to handle cusap API calls.
- [ ] Add smart pointers for cusap memory management.
- [ ] Add a way to handle cusap errors with tl:expected.

# libs/cupca_utils TODOS (To Complete)
- [ ] Add a way to handle cupca API calls.
- [ ] Add smart pointers for cupca memory management.
- [ ] Add a way to handle cupca errors with tl:expected.

# libs/holoflow TODOS (To Complete)
- [ ] Add a way to create a tree-like image processing pipeline.
- [ ] Add a model execution engine to execute the pipeline.
- [ ] Add a way to visualize the pipeline.
- [ ] Add a way to modify the pipeline while running.
- [ ] Add a way to save / load the pipeline.

# apps/holovibes TODOS (To Complete)

*See with project physician for a more detailed requirements list.*

# apps/holodev TODOS (To Complete)

*See with project physician for a more detailed requirements list.*