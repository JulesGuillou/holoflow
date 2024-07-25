load("//tools:copts.bzl", "common_copts")

cc_test(
    name = "aab_tests",
    srcs = ["aab_tests.cc"],
    copts = common_copts(),
    deps = [
        "//libs/cuaab:cuaab",
        "@googletest//:gtest_main",
    ],
)