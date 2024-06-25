load("//tools:copts.bzl", "common_copts")

cc_test(
    name = "holofile_reader_tests",
    srcs = ["holofile_reader_tests.cc"],
    copts = common_copts(),
    deps = [
        "//libs/holofile:holofile",
        "@googletest//:gtest_main",
    ],
)