load("//tools:copts.bzl", "common_copts")

cc_library(
    name = "batched_spsc_queue",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/*.hh"]),
    copts = common_copts(),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "batched_spsc_queue_tests",
    srcs = glob(["tests/*.cc"]),
    copts = common_copts(),
    deps = [
        ":batched_spsc_queue",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "batched_spsc_queue_benchmark",
    srcs = glob(["benchmarks/*.cc"]),
    copts = common_copts(),
    deps = [
        ":batched_spsc_queue",
        "@google_benchmark//:benchmark",
    ],
)
