# pympc-quadruped
A Python implementation about quadruped locomotion using convex model predictive control (MPC).

## Installation

### 1. Install Simulators
a. Mujoco
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

b. Isaac Gym

### 2. Create a Virtual Environment
a. Install virtualenv:
~~~
$ sudo apt install python3-virtualenv
~~~
or
~~~
$ pip install virtualenv
~~~

b. Create a virtual environment:
~~~
$ cd ${path-to-pympc-quadruped}
$ virtualenv --python /usr/bin/python3.8 pympc-env
~~~

### 3. Install Dependences
~~~
$ cd ${path-to-pympc-quadruped}
$ source ${path-to-pympc-quadruped}/pympc-env/bin/activate
(pympc-env)$ pip install --upgrade pip
(pympc-env)$ pip install -r requirements.txt 
~~~

## Sign Convention