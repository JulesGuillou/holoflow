load("//tools:copts.bzl", "common_copts")

cc_binary(
    name = "benchmarks",
    srcs = ["benchmarks.cc"],
    copts = common_copts(),
    deps = [
        "//libs/batched_spsc_queue:batched_spsc_queue",
        "@google_benchmark//:benchmark",
    ],
)