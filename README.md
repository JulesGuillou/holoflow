# Holoflow

**This project is still in early development and is not yet ready for use.**

Holoflow is an signal processing framework in C++/CUDA. It aims to provide an abstraction for the exectution of a model on a GPU.
The user describes the model as a tree where node are computational tasks and edges data dependencies. The framework then takes care of the execution of the model on the GPU.

## Documentation
Up-to-date documentation can be found in the at the [Holoflow documentation](https://julesguillou.github.io/holoflow/).

## Requirements
- C++20
- CUDA 12.0
- Bazel 7.0.0

## Layout
The project is divided into the following directories:
- `apps/` contains applications that use the Holoflow framework.
- `benchmarks/` contains benchmarks for the Holoflow framework.
- `docs/` contains documentation for the Holoflow framework.
- `examples/` contains examples for the Holoflow framework and the other libraries.
- `libs/` contains the Holoflow framework and a set of libraries that are used by the framework that will eventually be moved to a separate repository.
- `tests/` contains tests for the Holoflow framework.
- `third_party/` contains third party libraries that are used by the Holoflow framework. Note that some libraries are directly retrieved via bazel and are not stored in this directory.
- `tools` contains tools that are used by the build system.

## Building
The project uses Bazel as a build system. To build the project, run the following command:
```
bazel build //...
```
*See the [Bazel build documentation](https://docs.bazel.build/versions/main/build-ref.html) for more information.*

## Running tests
To run the tests, run the following command:
```
bazel test //...
```
*See the [Bazel test documentation](https://docs.bazel.build/versions/main/test-encyclopedia.html) for more information.*

## Running benchmarks
To run the benchmarks, run the following command:
```
bazel run //benchmarks/...
```
*See the [Bazel run documentation](https://docs.bazel.build/versions/main/command-line-reference.html#run) for more information.*

## Running examples
To run the examples, run the following command:
```
bazel run //examples/...
```
*See the [Bazel run documentation](https://docs.bazel.build/versions/main/command-line-reference.html#run) for more information.*

## Tips for development
You can generate the compilation database by running the following command:
```
bazel run @hedron_compile_commands//:refresh_all
```
*Note that you should have build the project first.*

## TODOS
You can find the list of todos in the [TODO.md](TODO.md) file. This list is not exhaustive and is subject to change.