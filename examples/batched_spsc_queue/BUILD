load("//tools:copts.bzl", "common_copts")

cc_binary(
    name = "multi_thread_example",
    srcs = ["multi_thread_example.cc"],
    copts = common_copts(),
    deps = [
        "//libs/batched_spsc_queue:batched_spsc_queue",
    ],
)