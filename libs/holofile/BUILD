load("//tools:copts.bzl", "common_copts")

cc_library(
    name = "holofile",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/*.hh"]),
    copts = common_copts(),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = ["//third_party:expected"],
)
