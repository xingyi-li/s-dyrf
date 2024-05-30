conda activate s-dyrf

SCENE=bouncingballs
STYLE=bouncingballs_colorful
python ref_pre.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=hellwarrior
STYLE=hellwarrior_spongebob
python ref_pre.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=hook
STYLE=hook_fire
python ref_pre.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=jumpingjacks
STYLE=jumpingjacks_pencil
python ref_pre.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=lego
STYLE=lego_starry_night
python ref_pre.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=mutant
STYLE=mutant_wave
python ref_pre.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=standup
STYLE=standup_autumn
python ref_pre.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=trex
STYLE=trex_paint
python ref_pre.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"