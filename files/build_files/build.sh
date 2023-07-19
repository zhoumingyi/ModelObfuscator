bazel build --jobs=14 //tensorflow/tools/pip_package:build_pip_package
# sleep 5
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg