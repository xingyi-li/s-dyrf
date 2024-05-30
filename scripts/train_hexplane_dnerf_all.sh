conda activate s-dyrf

SCENE=bouncingballs
python main.py config=config/dnerf/dnerf_${SCENE}.yaml

SCENE=hellwarrior
python main.py config=config/dnerf/dnerf_${SCENE}.yaml

SCENE=hook
python main.py config=config/dnerf/dnerf_${SCENE}.yaml

SCENE=jumpingjacks
python main.py config=config/dnerf/dnerf_${SCENE}.yaml

SCENE=lego
python main.py config=config/dnerf/dnerf_${SCENE}.yaml

SCENE=mutant
python main.py config=config/dnerf/dnerf_${SCENE}.yaml

SCENE=standup
python main.py config=config/dnerf/dnerf_${SCENE}.yaml

SCENE=trex
python main.py config=config/dnerf/dnerf_${SCENE}.yaml