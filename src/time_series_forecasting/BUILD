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
  ],
)
