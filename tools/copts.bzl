"""Common compiler options for all targets in all packages of the workspace."""

def common_copts():
    return select({
        "@platforms//os:windows": ["-std:c++20"],
        "@platforms//os:linux": ["-std=c++20"],
    })
