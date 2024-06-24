load("//tools:copts.bzl", "common_copts")

cc_library(
    name = "cudart_utils",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/*.hh"]),
    copts = common_copts(),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_cuda//:cuda_runtime",
        "//third_party:expected",
    ],
)
