tensorflow-python
---------------

Prerequisites
-------------

```
$ git clone --recursive https://github.com/pepeunlimited/tensorflow-python.git
```

Install bazelisk via homebrew  
```
$ brew install bazelisk
```

Test that bazel should be able to build iOS project with the Bazel
```
$ bazel build //HelloWorld/iOS:HelloWorld
```

Getting started
---------------

Bazel
-----

#### Run/Build/Test

##### src/lib

Build ```lib``` binary
```
$ bazel build //src/lib:lib
```

Run ```haberdasher_test``` test
```
$ bazel test //src/lib:haberdasher_test
```

##### src/time_series_forecasting

Run ```time_series_forecasting``` binary
```
$ bazel run //src/time_series_forecasting:main
```

Build ```time_series_forecasting``` binary
```
$ bazel build //src/time_series_forecasting:main
```

Run ```window_generator_test``` test
```
$ bazel test //src/time_series_forecasting:window_generator_test
```

unzip ```window_generator_test``` outputs.zip  
```
$ unzip bazel-testlogs/src/time_series_forecasting/dataset_test/test.outputs/outputs.zip -d src/time_series_forecasting/tests/output
```

Run ```dataset_test``` test  

Plot files are written to ```$SRCROOT/bazel-testlogs/src/time_series_forecasting/dataset_test/test.outputs/outputs.zip```
using the ```TEST_UNDECLARED_OUTPUTS_DIR``` environment variable.  

```
$ bazel test --test_output=all //src/time_series_forecasting:dataset_test
```

unzip ```dataset_test``` outputs.zip  
```
$ unzip bazel-testlogs/src/time_series_forecasting/dataset_test/test.outputs/outputs.zip -d src/time_series_forecasting/tests/output
```

Run ```model_test``` test
```
$ bazel test --test_output=all //src/time_series_forecasting:model_test
```

unzip ```model_test``` outputs.zip  
```
$ unzip bazel-testlogs/src/time_series_forecasting/model_test/test.outputs/outputs.zip -d src/time_series_forecasting/tests/output
```

Run ```example_test``` test
```
$ bazel test //src/time_series_forecasting:example_test
```

Run all tests
```
$ bazel test //src/time_series_forecasting:all
```

##### tests/another_test

Run ```unittest_test```
```
$ bazel test //tests/another_test:unittest_test
```

#### Requirements.txt  

Initialize pip requires an a empty file. Name is defined in the ```MODULE.bazel```
and ```BUILD``` files. See examples inside ```src/time-series-forecasting/BUILD```. Actual dependencies
are defined in ```requirements.in```.

```
$ touch requirements_lock.txt
```

After that run following command
```
$ bazel run //:requirements.update
```

Clean  
```
$ bazel clean --expunge
```

Documentation & Links
---------------------

[`This documentation is collected from rulesets in the bazelbuild GitHub org`](https://docs.aspect.build/)
<br/>
[`bazelbuild/rules_python`](https://github.com/bazelbuild/rules_python/blob/main/docs/sphinx/getting-started.md)  
[`bazel.build/python`](https://bazel.build/reference/be/python)  
[`bazel.build/python/bzlmod`](https://bazel.build/reference/be/pythohttps://github.com/bazelbuild/rules_python/blob/main/examples/bzlmod/BUILD.bazeln)  
<br/>
[`bazel.build/command-line-reference`](https://bazel.build/reference/command-line-reference)  
[`Bzlmod Migration Guide`](https://bazel.build/external/migration)  
<br/>
[`Getting Started With Testing in Python`](https://realpython.com/python-testing/)  
[`Structuring Your Project`](https://docs.python-guide.org/writing/structure/)  
<br/>
[`How to search code with Sourcegraph — a cheat sheet`](https://sourcegraph.com/blog/how-to-search-cheat-sheet)  
[`BUILD and py_test search`](https://sourcegraph.com/search?q=context:global+file:BUILD%24+AND+file:has.content%28py_test%29)  
[`file:pyrightconfig\.json$ AND file:has.content(reportUnknownMemberType)`](https://sourcegraph.com/search?q=context:global+file:pyrightconfig%5C.json%24+AND+file:has.content%28reportUnknownMemberType%29)  
<br/>
[`Understanding-LSTMs`](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
[`Simple LSTM`](https://github.com/nicodjimenez/lstm)


License
-------

**tensorflow-python** is released under the MIT license. See `LICENSE` for details.
