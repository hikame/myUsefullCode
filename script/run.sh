adb push test/test_x2 /data/local/tmp/vasr
adb push libosr.so /data/local/tmp/vasr
adb shell export LD_LIBRARY_PATH=/data/local/tmp/vasr
adb shell LD_LIBRARY_PATH="/data/local/tmp/vasr" /data/local/tmp/vasr/test_x2
#adb pull /data/local/tmp/vasr/high_resolution_x2.jpg
