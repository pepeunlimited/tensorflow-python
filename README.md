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

```
$ which bazel
```

Since Homebrew v3.0.0, the default prefix is different depending on the chip architecture. The defaults are the following:

on Apple silicon
```
/opt/homebrew
```

on Intel
```
/usr/local
```

Bazel
-----

Clean  
```
$ bazel clean --expunge
```

Documentation & Links
---------------------

[`This documentation is collected from rulesets in the bazelbuild GitHub org`](https://docs.aspect.build/)
<br/>
[`bazelbuild/rules_apple`](https://github.com/bazelbuild/rules_apple/tree/master/doc)  
[`bazelbuild/rules_swift`](https://github.com/bazelbuild/rules_swift/tree/master/doc)  
[`bazelbuild/apple_support`](https://github.com/bazelbuild/apple_support/tree/master/doc)  
[`bazelbuild/bazel-skylib`](https://github.com/bazelbuild/bazel-skylib/tree/main/docs)  
[`bazel.build/command-line-reference`](https://bazel.build/reference/command-line-reference)  
[`buildbuddy-io/rules_xcodeproj`](https://github.com/buildbuddy-io/rules_xcodeproj/tree/main/docs)
<br/>
[`Bazel Tutorial: Build an iOS App`](https://bazel.build/tutorials/ios-app)  
[`Migrating from Xcode to Bazel`](https://bazel.build/migrate/xcode)  
[`Building with Bazel`](https://www.raywenderlich.com/31558158-building-with-bazel/)  
[`ios_and_bazel_at_reddit_a_journey`](https://www.reddit.com/r/RedditEng/comments/syz5dw/ios_and_bazel_at_reddit_a_journey/)  
[`migrating-ios-project-to-bazel-a-real-world-experience`](https://liuliu.me/eyes/migrating-ios-project-to-bazel-a-real-world-experience/)  
<br/>
[`google-mediapipe-examples-ios`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios)  
[`Telegram-iOS`](https://github.com/TelegramMessenger/Telegram-iOS)  
[`liuliu/dflat`](https://github.com/liuliu/dflat)  
[`wendyliga/simple_bazel`](https://github.com/wendyliga/simple_bazel)  
[`TulsiGeneratorIntegrationTests`](https://github.com/bazelbuild/tulsi/tree/master/src/TulsiGeneratorIntegrationTests/Resources)  
[`iOS Dynamic vs. Static Library / Framework`](https://gist.github.com/SheldonWangRJT/78c9bd3b98488487c59a6a4a9c35162c)  
<br/>


License
-------

**tensorflow-python** is released under the MIT license. See `LICENSE` for details.
