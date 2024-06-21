load("//tools:copts.bzl", "common_copts")

cc_test(
    name = "capacity_tests",
    srcs = ["capacity_tests.cc"],
    copts = common_copts(),
    deps = [
        "//libs/batched_spsc_queue:batched_spsc_queue",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "multithread_tests",
    srcs = ["multithread_tests.cc"],
    copts = common_copts(),
    deps = [
        "//libs/batched_spsc_queue:batched_spsc_queue",
        "@googletest//:gtest_main",
    ],
)