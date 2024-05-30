EXPNAME=baseline

SCENE=coffee_martini
STYLE=coffee_martini_colorful
cp dataset/neural_3D/${SCENE}/poses_bounds.npy stylize/neural3D_NDC/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python main.py config="config/nv3d/nv3d_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/neural3D_NDC/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_time=True

SCENE=cook_spinach
STYLE=cook_spinach_paint
cp dataset/neural_3D/${SCENE}/poses_bounds.npy stylize/neural3D_NDC/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python main.py config="config/nv3d/nv3d_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/neural3D_NDC/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_time=True

SCENE=cut_roasted_beef
STYLE=cut_roasted_beef_wave
cp dataset/neural_3D/${SCENE}/poses_bounds.npy stylize/neural3D_NDC/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python main.py config="config/nv3d/nv3d_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/neural3D_NDC/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_time=True

SCENE=flame_salmon
STYLE=flame_salmon_spongebob
cp dataset/neural_3D/${SCENE}/poses_bounds.npy stylize/neural3D_NDC/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python main.py config="config/nv3d/nv3d_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/neural3D_NDC/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_time=True

SCENE=flame_steak
STYLE=flame_steak_starry_night
cp dataset/neural_3D/${SCENE}/poses_bounds.npy stylize/neural3D_NDC/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python main.py config="config/nv3d/nv3d_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/neural3D_NDC/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_time=True

SCENE=sear_steak
STYLE=sear_steak_pencil
cp dataset/neural_3D/${SCENE}/poses_bounds.npy stylize/neural3D_NDC/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python main.py config="config/nv3d/nv3d_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/neural3D_NDC/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_time=True

