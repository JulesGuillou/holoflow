load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("//tools:copts.bzl", "common_copts")

cuda_library(
    name = "cuaab",
    srcs = glob(["src/*.cc", "src/*.cu"]),
    hdrs = glob(["include/*.hh"]),
    copts = common_copts(),
    includes = ["include"],
    visibility = ["//visibility:public"],
)