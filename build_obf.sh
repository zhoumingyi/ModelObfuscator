python obfuscation.py --model_name=fruit --extra_layer=30 --shortcut=30
# ----------------------------------------------
# uncomment the following lines if you want to test the uint8 model (but we currently only support limited ops for uint8 model)
# flatc -t schema.fbs -- obf_model.tflite
# python modify_tflite.py
# flatc -b ./schema.fbs obf_model.json
# ----------------------------------------------
python -m pip uninstall -y tensorflow
python -m pip install /tmp/tensorflow_pkg/tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl
python test_model.py --model_name=fruit
