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

#### Run & Build

Run binary
```
$ bazel run //src/time_series_forecasting:main
```

Build binary
```
$ bazel run //src/time_series_forecasting:main
```

#### Requirements.txt  

Initialize pip requirements file requires to create empty file. Name is defined in the ```MODULE.bazel```
and ```BUILD``` files. See examples inside ```src/time-series-forecasting/BUILD```. Actual dependencies
is defined in ```requirements.in```.

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
<br/>

[`bazel.build/command-line-reference`](https://bazel.build/reference/command-line-reference)  
[`Bzlmod Migration Guide`](https://bazel.build/external/migration)  
<br/>

License
-------

**tensorflow-python** is released under the MIT license. See `LICENSE` for details.
