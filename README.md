

Add lines below to lab/BUILD file.

```
# Setting for Epsodic control
py_binary(
    name = "ec_train",
    srcs = ["episodic_control/train.py"],
    data = [":deepmind_lab.so"],
    main = "episodic_control/train.py"
)
```

To run training, run command below in 'lab' directory.

```
$ bazel run :ec_train --define headless=osmesa
```
