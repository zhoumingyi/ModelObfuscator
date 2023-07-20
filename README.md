# ModelObfuscator

This is protptype tool of paper: Modelobfuscator: Obfuscating Model Information to Protect Deployed ML-Based Systems on ISSTA2023.

## 1*. Preparation A: get the environment by Docker (recommend)

(0) Download the Docker Image:

```
docker pull anonymousauthor000/code275:v3.1
```

Note that if it cause permission errors, please try: 

```
sudo docker pull anonymousauthor000/code2536:v2
```

(1) Enter the environment:

```
docker run -i -t anonymousauthor000/code275:v3.1 /bin/bash
```

Note that if it cause permission errors, please try: 

```
docker run -i -t anonymousauthor000/code2536:v2 /bin/bash
```

Enter the project:

```
cd code275/
```

(2) Activate the conda environment: 

```
conda activate code275
```

## 1*. Preparation B: build the environment

(0) Download the code:

```
git clone https://github.com/zhoumingyi/ModelObfuscator.git
cd ModelObfuscator
```

(1) The dependency can be found in `environment.yml`. To create the conda environment:

```
conda env create -f environment.yml
conda activate code275
```

Install the Flatbuffer:

```
conda install -c conda-forge flatbuffers
```

(if no npm) install the npm:

```
sudo apt-get install npm
```

Install the jsonrepair:

```
npm install -g jsonrepair
```

Note that the recommend version of gcc and g++ is 9.4.0.


(2) Download the source code of the TensorFlow. Here we test our tool on v2.9.1.

```
wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.zip
```

Unzip the file:

```
unzip v2.9.1
```

(3) Download the Bazel:

```
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
```

You can test the Bazel:

```
which bazel
```

It should return:

```
# in ubuntu
/usr/local/bin/bazel
```

(4) Configure the build:

```
cd tensorflow-2.9.1/
./configure
cd ..
```

You can use the default setting (just type Return/Enter for every option).

(5) Copy the configurations and script to the source code:  

```
cp ./files/kernel_files/* ./tensorflow-2.9.1/tensorflow/lite/kernels/
cp ./files/build_files/build.sh ./tensorflow-2.9.1/
```

Note that you can mofify the maximal number of jobs in the 'build.sh' script. Here I set it as `--jobs=14`. 

## 2. Test

(1) Build the obfuscation model:

```
bash build_obf.sh
```

Note that you can modify the test model and obfuscation parameters in the script. The obfuscated model is saved as the 'obf_model.tflite'.
