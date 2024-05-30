conda activate fspbt

SCENE=coffee_martini
STYLE=coffee_martini_colorful
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=cook_spinach
STYLE=cook_spinach_paint
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=cut_roasted_beef
STYLE=cut_roasted_beef_wave
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=flame_salmon
STYLE=flame_salmon_spongebob
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=flame_steak
STYLE=flame_steak_starry_night
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=sear_steak
STYLE=sear_steak_pencil
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time