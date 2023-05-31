# pympc-quadruped
A Python implementation about quadruped locomotion using convex model predictive control (MPC).

![image](https://github.com/yinghansun/pympc-quadruped/blob/main/doc/results/trotting10_mujoco.gif)

## Installation

### 1. Create a Virtual Environment
#### a. Install virtualenv:
~~~
$ sudo apt install python3-virtualenv
~~~
or
~~~
$ pip install virtualenv
~~~

#### b. Create a virtual environment:
~~~
$ cd ${path-to-pympc-quadruped}
$ virtualenv --python /usr/bin/python3.8 pympc-env
~~~

### 2. Install Simulators
#### a. Mujoco
- Create a folder named `.mujoco` in your home directory: `$ mkdir ~/.mujoco`.
- Download Mujoco library from https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz. Extract and move it to the `.mujoco` folder.
- Add the following to your `.bashrc` file:
    ~~~
    export LD_LIBRARY_PATH=/home/${your-usr-name}/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export PATH="$LD_LIBRARY_PATH:$PATH"
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    ~~~
    Do not forget to run `$ source ~/.bashrc`.
- Test if the installation is successfully finished:
    ~~~
    $ cd ~/.mujoco/mujoco210/bin
    $ ./simulate ../model/humanoid.xml
    ~~~

#### b. Isaac Gym
- Download Isaac Gym Preview Release from this [website](https://developer.nvidia.com/isaac-gym). 
- The tutorial for installation is in the `./isaacgym/docs/install.html`. **I recommand the user to install it in the previous virtual environment**. 
    ~~~
    $ cd ${path-to-issacgym}/python
    $ source ${path-to-pympc-quadruped}/pympc-env/bin/activate
    (pympc-env)$ pip install -e .
    ~~~
- Then you can trying to run examples in `./isaacgym/python/examples`. Note that if you follow the instructions above, you need to run the examples in the virtual environments.
    ~~~
    $ cd ${path-to-isaacgym}/python/examples
    $ source ${path-to-pympc-quadruped}/pympc-env/bin/activate
    (pympc-env)$ python 1080_balls_of_solitude.py
    ~~~
- For troubleshooting, check `./isaacgym/docs/index.html`

**Note**: If you meet the following issue like
`ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory` when running the examples, you can try 
`export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib` before executing your python script. If you are not using conda, the path shoud be `/path/to/libpython/directory`.

### 3. Install Dependences

**a) Install Pinocchio** 

Pinocchio provides the state-of-the-art rigid body kinematics and dynamic algorithms. You could follow [this link](https://stack-of-tasks.github.io/pinocchio/download.html) to install Pinocchio.

**b) Install other dependences**

~~~
$ cd ${path-to-pympc-quadruped}
$ source ${path-to-pympc-quadruped}/pympc-env/bin/activate
(pympc-env)$ pip install --upgrade pip
(pympc-env)$ pip install -r requirements.txt 
~~~

## Sign Convention