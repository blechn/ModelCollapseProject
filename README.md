# Model collapse investigation
As a final project for the lecture "Generative Neural Networks for the Sciences" at Heidelberg University in the WS25/26.  
This repository includes the source code for the project of me. It is merged into the shared repository at [todo: link].  

## To run the experiments
you can run the premade bash scripts for the models. This will run all experiments for the respective model, or all experiments for all models if you run `./run_all.sh`. Note that this can take a lot of time, around one day on my PC.  

Alternatively, if you want to run only a specific experiment, you can do:
```
python -m src.routines.full_experiment \
            --modelcls [realnvp, flowmatching, pixelcnn] \
            --experiment [full, replace, add] \
            --collapse_epochs 10 \
            --max_epochs 10 \
            --add_percentage 0.2 \
            --replace_percentage 0.2 \
            [--fashion]
```

For a description of the command line parameters, run `python -m src.routines.full_experiment --help`.  


## License
This source code is licensed under the AGPLv3.0. See the LICENSE.txt file for more information.
