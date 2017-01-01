

Add lines below to lab/BUILD file.

```
# Setting for Epsodic control
py_binary(
    name = "ec_train",
    srcs = ["episodic_control/train.py"],
    data = [":deepmind_lab.so"],
    main = "episodic_control/train.py"
)

py_test(
    name = "ec_test",
    srcs = ["episodic_control/test.py"],
    main = "episodic_control/test.py",
    deps = [":ec_train"],  
)

```

To run training, run command below in 'lab' directory.

```
$ bazel run :ec_train --define headless=osmesa
```

To run test,

```
$ bazel run :ec_test --define headless=osmesa
```
