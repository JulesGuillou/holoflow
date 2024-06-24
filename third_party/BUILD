load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "expected_srcs",
    srcs = glob(["expected/**"]),
)

cmake(
    name = "expected",
    lib_source = ":expected_srcs",
    out_headers_only = True,
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
)