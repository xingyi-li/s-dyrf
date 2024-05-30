conda activate s-dyrf

SCENE=coffee_martini
python main.py config=config/nv3d/nv3d_${SCENE}.yaml

SCENE=cook_spinach
python main.py config=config/nv3d/nv3d_${SCENE}.yaml

SCENE=cut_roasted_beef
python main.py config=config/nv3d/nv3d_${SCENE}.yaml

SCENE=flame_salmon
python main.py config=config/nv3d/nv3d_${SCENE}.yaml

SCENE=flame_steak
python main.py config=config/nv3d/nv3d_${SCENE}.yaml

SCENE=sear_steak
python main.py config=config/nv3d/nv3d_${SCENE}.yaml
