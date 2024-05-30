conda activate s-dyrf

SCENE=coffee_martini
python main_finetune_no_view.py systems.ckpt="log/nv3d/${SCENE}/nv3d.th" config="config/nv3d/nv3d_${SCENE}.yaml"

SCENE=cook_spinach
python main_finetune_no_view.py systems.ckpt="log/nv3d/${SCENE}/nv3d.th" config="config/nv3d/nv3d_${SCENE}.yaml"

SCENE=cut_roasted_beef
python main_finetune_no_view.py systems.ckpt="log/nv3d/${SCENE}/nv3d.th" config="config/nv3d/nv3d_${SCENE}.yaml"

SCENE=flame_salmon
python main_finetune_no_view.py systems.ckpt="log/nv3d/${SCENE}/nv3d.th" config="config/nv3d/nv3d_${SCENE}.yaml"

SCENE=flame_steak
python main_finetune_no_view.py systems.ckpt="log/nv3d/${SCENE}/nv3d.th" config="config/nv3d/nv3d_${SCENE}.yaml"

SCENE=sear_steak
python main_finetune_no_view.py systems.ckpt="log/nv3d/${SCENE}/nv3d.th" config="config/nv3d/nv3d_${SCENE}.yaml"
