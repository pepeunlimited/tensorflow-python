#
#  BUILD
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

load("@pypi//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "lib",
    srcs = [
        "haberdasher.py"
    ],
    visibility = ["//visibility:public"],
    deps = [],
)

py_test(
    name = "haberdasher_test",
    srcs = [
        "tests/haberdasher_test.py"
    ],
    main = "haberdasher_test.py",
    timeout = "short",
    deps = [
        "//src/lib:lib",
    ],
)
