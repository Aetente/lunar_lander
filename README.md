# lunar_lander
To train the model run dqn.py, double_dqn1.py or double_dqn2.py

You can ran it with path to the state of trained model, if you have it. Otherwise it just trains new model.

For example:

  python dqn.py checkpoint1.pth
 
You can play the trained model (render evaluation of the model) with play.py.
 
You need to give the path to the trained model state in the arguments.

For example:

  python play.py checkpoint2.pth
