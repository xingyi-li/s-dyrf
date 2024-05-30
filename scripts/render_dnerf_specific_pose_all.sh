SCENE=bouncingballs
STYLE=bouncingballs_colorful
python render_dnerf_specific_pose.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=hellwarrior
STYLE=hellwarrior_spongebob
python render_dnerf_specific_pose.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=hook
STYLE=hook_fire
python render_dnerf_specific_pose.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=jumpingjacks
STYLE=jumpingjacks_pencil
python render_dnerf_specific_pose.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=lego
STYLE=lego_starry_night
python render_dnerf_specific_pose.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=mutant
STYLE=mutant_wave
python render_dnerf_specific_pose.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=standup
STYLE=standup_autumn
python render_dnerf_specific_pose.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"

SCENE=trex
STYLE=trex_paint
python render_dnerf_specific_pose.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"