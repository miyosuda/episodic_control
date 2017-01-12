## How to run
First, dowload and install DeepMind Lab
```
$ git clone https://github.com/deepmind/lab.git
```
Then build it following the build instruction. 
https://github.com/deepmind/lab/blob/master/docs/build.md

Clone this repo in lab directory.
```
$ cd lab
$ git clone https://github.com/miyosuda/episodic_control.git
```

Add this bazel instrution at the end of `lab/BUILD` file

```
package(default_visibility = ["//visibility:public"])
```

To run training, run command below in 'lab' directory.

```
$ bazel run //episodic_control:train --define headless=osmesa
```

To run test,

```
$ bazel run //episodic_control:test --define headless=osmesa
```
