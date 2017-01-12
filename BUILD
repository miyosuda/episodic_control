package(default_visibility = ["//visibility:public"])

py_binary(
    name = "train",
    srcs = ["train.py"],
    data = ["//:deepmind_lab.so"],
    main = "train.py"
)

py_test(
    name = "test",
    srcs = ["test.py"],
    main = "test.py",
    deps = [":train"],
)
