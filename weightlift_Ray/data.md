source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolo_ov
unset PYTHONPATH
cd /home/robotis/Ray_wk/src/fira_basketball/scripts
python3 weightlift/detector.py


預覽畫面: http://127.0.0.1:8080/





source /opt/ros/kinetic/setup.bash
source ~/Ray_wk/devel/setup.bash
export PYTHONPATH=$PYTHONPATH:/home/robotis/Ray_wk/src/fira_basketball/scripts
cd ~/Ray_wk/src/fira_basketball/scripts/weightlift
python op3_weightlifting_v4.py
