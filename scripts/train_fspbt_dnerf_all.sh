conda activate fspbt

SCENE=bouncingballs
STYLE=bouncingballs_colorful
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=hellwarrior
STYLE=hellwarrior_spongebob
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=hook
STYLE=hook_fire
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=jumpingjacks
STYLE=jumpingjacks_pencil
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=lego
STYLE=lego_starry_night
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=mutant
STYLE=mutant_wave
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=standup
STYLE=standup_autumn
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time

SCENE=trex
STYLE=trex_paint
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time