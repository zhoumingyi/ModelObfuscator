#Original_code
# bazel build --jobs=14 //tensorflow/tools/pip_package:build_pip_package
# sleep 5
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

#Modified code

bazel build --jobs=14 //tensorflow/tools/pip_package:build_pip_package
echo "Hello, generating a package unpacking successfull "
echo $PWD
mkdir -p /tmp/tensorflow_pkg && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
echo "Hello, package creation is successfull "

