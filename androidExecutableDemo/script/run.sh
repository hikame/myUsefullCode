adb push test/test_demo /data/local/tmp/
adb shell export LD_LIBRARY_PATH=/data/local/tmp/vasr
adb shell LD_LIBRARY_PATH="/data/local/tmp/" /data/local/tmp/test_demo
