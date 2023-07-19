python obfuscation.py --model_name=fruit --extra_layer=30 --shortcut=30
flatc -t schema.fbs -- obf_model.tflite
python modify_tflite.py
flatc -b ./schema.fbs obf_model.json
pip uninstall -y tensorflow
pip install /tmp/tensorflow_pkg/tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl
python test_model.py --model_name=fruit
