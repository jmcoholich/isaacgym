# This script is for automatically generating videos on a remote server with a display.
# Must be run from this directory
set -ex

H_model=123750
F_model=1234750
H_model_fname="H_model_video"
F_model_fname="F_model_video"
TIMEOUT=20_000
START_AFTER=50
# don't forget, I think we need a gpu with a display attached for the camera rendering to work properly

# time how long this script takes in minutes
start_time=$(date +%s)
# training env
python rlg_train.py --play --checkpoint $H_model --timeout $TIMEOUT --plot_values --des_dir 0.0 --des_dir_coef 50.0 --save_images --start_after $START_AFTER --file_prefix $H_model_fname  --headless --device_id 0 --seed 0

python rlg_train.py --play --checkpoint $F_model --timeout $TIMEOUT --save_images --file_prefix $F_model_fname  --headless --device_id 0 --seed 0

python make_video.py train 'Terrain: 90% Infill, 5cm Height Variance' 15
python make_video.py train 'Terrain: 90% Infill, 5cm Height Variance' 30
python make_video.py train 'Terrain: 90% Infill, 5cm Height Variance' 60

# flatground (100 infill and no height var)
python rlg_train.py --play --checkpoint $H_model --timeout $TIMEOUT --plot_values --des_dir 0.0 --des_dir_coef 50.0 --save_images --start_after $START_AFTER --file_prefix $H_model_fname  --headless --device_id 0 --seed 0 --no_ss

python rlg_train.py --play --checkpoint $F_model --timeout $TIMEOUT --save_images --file_prefix $F_model_fname  --headless --device_id 0 --seed 0 --no_ss

python make_video.py flatground 'Flatground' 15
python make_video.py flatground 'Flatground' 30
python make_video.py flatground 'Flatground' 60

# hardest env (80 infill, 10cm height var)
python rlg_train.py --play --checkpoint $H_model --timeout $TIMEOUT --plot_values --des_dir 0.0 --des_dir_coef 50.0 --save_images --start_after $START_AFTER --file_prefix $H_model_fname  --headless --device_id 0 --seed 0 --add_ss --ss_infill 0.8 --ss_height_var 0.075

python rlg_train.py --play --checkpoint $F_model --timeout $TIMEOUT --save_images --file_prefix $F_model_fname  --headless --device_id 0 --seed 0 --add_ss --ss_infill 0.8 --ss_height_var 0.075

python make_video.py hardest 'Terrain: 80% Infill, 7.5cm Height Variance' 15
python make_video.py hardest 'Terrain: 80% Infill, 7.5cm Height Variance' 30
python make_video.py hardest 'Terrain: 80% Infill, 7.5cm Height Variance' 60

end_time=$(date +%s)
echo "Total time: $((end_time - start_time)) seconds"
