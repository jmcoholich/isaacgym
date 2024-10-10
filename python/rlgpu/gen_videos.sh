# This script is for automatically generating videos on a remote server with a display.
# Must be run from this directory


H_model=123750
F_model=1234750
H_model_fname="H_model_video"
# don't forget, I think we need a gpu with a display attached for the camera rendering to work properly

python rlg_train.py --play --checkpoint $H_model --ws 7 --timeout 2000 --plot_values --des_dir_coef 0.0 --des_dir_coef 50.0 --save_images --start_after 300 --file_prefix $H_model_fname  --headless  --device_id 0

cd ../..

python make_video.py --output_name=H_model_video --file_prefix $H_model_fname  --no_plots

cd python/rlgpu


# export DISPLAY=:1

# models=(211130223242990594 211130223308996394 211130223315201609 211130224435963959 211130223347357130)
# names=("left_bias" "fwd_bias" "right_bias" "back_bias" "regular_dist")


# for i in ${!names[@]}
# do
#     echo run_number:    $i
#     echo name:          ${names[$i]}
#     echo model:         ${models[$i]}
#     echo

#     python rlg_train.py --play --checkpoint ${models[$i]} --ws 7 --timeout 2000 --plot_values --des_dir_coef 0.0 --cfg_env_override 9_trot_in_place --save_images --exit_after 1800 --start_after 300 --file_prefix ${names[$i]}

#     cd ../..

#     python make_video.py --output_name=${names[$i]} --file_prefix ${names[$i]}

#     cd python/rlgpu

# done

# exit

# script for showing different directions

# export DISPLAY=:1

# names=("fwd" "fwd-left" "left" "back-left" "back" "back-right" "right" "fwd-right" "in-place")
# des_dirs=(0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 0.0)
# des_dir_coefs=(0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.0)

# for i in ${!names[@]}
# do
#     echo run_number:    $i
#     echo name:          ${names[$i]}
#     echo des_dir:       ${des_dirs[$i]}
#     echo des_dir_coef:  ${des_dir_coefs[$i]}
#     echo

#     python rlg_train.py --checkpoint 211104024526566297 --ws 7 --play --plot_values --exit_after 2000 --save_images --timeout 2000 --cfg_env_override trot_in_place_no_contact_info --des_dir ${des_dirs[$i]} --des_dir_coef ${des_dir_coefs[$i]}

#     cd ../..

#     python make_video.py --output_name="${names[$i]} ${des_dir_coefs[$i]} coef"

#     cd python/rlgpu

# done

# exit
