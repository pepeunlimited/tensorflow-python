#
#  BUILD
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

load("@rules_python//python:defs.bzl", "py_binary")
load("@pypi//:requirements.bzl", "requirement")

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        requirement("requests"),
        requirement("tensorflow"),
        requirement("pandas")
    ],
)

py_library(
    name = "lib",
    srcs = [
        "window_generator.py"
    ],
    visibility = ["//visibility:public"],
    deps = [
        requirement("requests"),
        requirement("tensorflow"),
        requirement("pandas")
    ],
)

py_test(
    name = "window_generator_test",
    srcs = [
        "tests/window_generator_test.py",
    ],
    main = "window_generator_test.py",
    timeout = "short",
    deps = [
        "//src/time_series_forecasting:lib",
    ],
)

py_test(
    name = "example_test",
    srcs = [
        "tests/example_test.py"
    ],
    main = "example_test.py",
    timeout = "short",
    deps = [
        "//src/time_series_forecasting:lib",
    ],
)
