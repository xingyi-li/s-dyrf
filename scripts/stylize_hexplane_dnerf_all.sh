EXPNAME=baseline

SCENE=bouncingballs
STYLE=bouncingballs_colorful
cp dataset/dnerf/${SCENE}/transforms_train.json stylize/dnerf/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python main.py config="config/dnerf/dnerf_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/dnerf/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_pose=True render_specific_time=True

SCENE=hellwarrior
STYLE=hellwarrior_spongebob
cp dataset/dnerf/${SCENE}/transforms_train.json stylize/dnerf/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python main.py config="config/dnerf/dnerf_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/dnerf/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_pose=True render_specific_time=True

SCENE=hook
STYLE=hook_fire
cp dataset/dnerf/${SCENE}/transforms_train.json stylize/dnerf/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python main.py config="config/dnerf/dnerf_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/dnerf/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_pose=True render_specific_time=True

SCENE=jumpingjacks
STYLE=jumpingjacks_pencil
cp dataset/dnerf/${SCENE}/transforms_train.json stylize/dnerf/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python main.py config="config/dnerf/dnerf_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/dnerf/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_pose=True render_specific_time=True

SCENE=lego
STYLE=lego_starry_night
cp dataset/dnerf/${SCENE}/transforms_train.json stylize/dnerf/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python main.py config="config/dnerf/dnerf_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/dnerf/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_pose=True render_specific_time=True

SCENE=mutant
STYLE=mutant_wave
cp dataset/dnerf/${SCENE}/transforms_train.json stylize/dnerf/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python main.py config="config/dnerf/dnerf_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/dnerf/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_pose=True render_specific_time=True

SCENE=standup
STYLE=standup_autumn
cp dataset/dnerf/${SCENE}/transforms_train.json stylize/dnerf/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python main.py config="config/dnerf/dnerf_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/dnerf/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_pose=True render_specific_time=True

SCENE=trex
STYLE=trex_paint
cp dataset/dnerf/${SCENE}/transforms_train.json stylize/dnerf/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python main.py config="config/dnerf/dnerf_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/dnerf/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_pose=True render_specific_time=True
