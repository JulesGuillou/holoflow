load("//tools:copts.bzl", "common_copts")

cc_test(
    name = "macro_tests",
    srcs = ["macro_tests.cc"],
    copts = common_copts(),
    deps = [
        "//libs/cudart_utils:cudart_utils",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "deleter_tests",
    srcs = ["deleter_tests.cc"],
    copts = common_copts(),
    deps = [
        "//libs/cudart_utils:cudart_utils",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "error_category_tests",
    srcs = ["error_category_tests.cc"],
    copts = common_copts(),
    deps = [
        "//libs/cudart_utils:cudart_utils",
        "@googletest//:gtest_main",
    ],
)