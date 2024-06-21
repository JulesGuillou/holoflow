load("//tools:copts.bzl", "common_copts")

cc_library(
    name = "batched_spsc_queue",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/*.hh"]),
    copts = common_copts(),
    includes = ["include"],
    visibility = ["//visibility:public"],
)