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
        requirement("pandas"),
        requirement("pandas-stubs"),
        requirement("matplotlib"),
        requirement("seaborn")
    ],
)

py_library(
    name = "lib",
    srcs = [
        "window_generator.py",
        "dataset.py"
    ],
    visibility = ["//visibility:public"],
    deps = [
        requirement("requests"),
        requirement("tensorflow"),
        requirement("pandas"),
        requirement("pandas-stubs"),
        requirement("matplotlib"),
        requirement("seaborn")
    ],
)

py_test(
    name = "window_generator_test",
    srcs = [
        "tests/window_generator_test.py",
    ],
    main = "tests/window_generator_test.py",
    timeout = "short",
    deps = [
        "//src/time_series_forecasting:lib",
    ],
)

# @see data files https://stackoverflow.com/a/42307073/3913343

filegroup(
    name = "data_dataset",
    srcs = glob([
        "tests/input/*.txt"
    ]),
)

py_test(
    name = "dataset_test",
    srcs = [
        "tests/dataset_test.py",
    ],
    main = "tests/dataset_test.py",
    timeout = "short",
    deps = [
        "//src/time_series_forecasting:lib",
    ],
    data = [
        "//src/time_series_forecasting:data_dataset"
    ],
)

py_test(
    name = "example_test",
    srcs = [
        "tests/example_test.py"
    ],
    main = "tests/example_test.py",
    timeout = "short",
    deps = [
        "//src/time_series_forecasting:lib",
    ],
)
