conda activate s-dyrf

SCENE=coffee_martini
STYLE=coffee_martini_colorful
python ref_pre.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"

SCENE=cook_spinach
STYLE=cook_spinach_paint
python ref_pre.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"

SCENE=cut_roasted_beef
STYLE=cut_roasted_beef_wave
python ref_pre.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"

SCENE=flame_salmon
STYLE=flame_salmon_spongebob
python ref_pre.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"

SCENE=flame_steak
STYLE=flame_steak_starry_night
python ref_pre.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"

SCENE=sear_steak
STYLE=sear_steak_pencil
python ref_pre.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"