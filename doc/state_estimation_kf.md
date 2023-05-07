# State Estimation - Linear Kalman Filter Approach

**Reference**: 

[1] Bledt G, Powell M J, Katz B, et al. Mit cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot[C]//2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018: 2245-2252. [[PDF]](https://dspace.mit.edu/bitstream/handle/1721.1/126619/IROS.pdf?sequence=2&isAllowed=y)

[2] Bloesch M, Hutter M, Hoepflinger M A, et al. State estimation for legged robots-consistent fusion of leg kinematics and IMU[J]. Robotics, 2013, 17: 17-24. [[PDF]](https://infoscience.epfl.ch/record/181040/files/Bloesch-Siegwart_State__IMU_2012.pdf)

[3] Mahony R, Hamel T, Pflimlin J M. Nonlinear complementary filters on the special orthogonal group[J]. IEEE Transactions on automatic control, 2008, 53(5): 1203-1218. [[PDF]](https://hal.archives-ouvertes.fr/hal-00488376/file/2007_Mahony.etal_TAC-06-396_v3.pdf)

## 1. Introduction

Cheetah 3 estimates its body states through a two-stage sensor fusion algorithm that *decouples estimation of body orientation from estimation of the body position and velocity* [1].

## 2. 1st Stage: Orientation Filter

The first stage of the state estimation employs an orientation filter using both the IMU gyro and accelerometer readings. The main idea of the filter is that the gyro provides an accurate reading of the high-frequency orientation dynamics, whereas the presence of a gravity bias on the accelerometer allows it to de-drift the estimate at a comparatively lower frequency [3].

The filter updates the estimation of orientation according to

$$
\hat{R}_{b,k+1}^o = \hat{R}_{b,k}^o [\omega_{b,k}^b + \kappa \omega_{\text{corr},k}]^{\times}
$$

where $\kappa > 0$ is a correction gain and $\omega_{\text{corr}}$ is a correction angular velocity to align the accelerometer reading $a_b$ with its gravity bias

$$
\omega_{\text{corr},k} = \frac{a_{b,k}^b}{||a_{b,k}^b||}\times (\hat{R}_{b,k}^o)^\text{T}\mathbf{e}_z
$$

The time constant of the de-drifting from this term can be approximated by $\kappa^{-1}$. In practice $\kappa$ is heuristically decreased during highly-dynamic portions of the gait where $||a_b|| \gg g$ with

$$
\kappa = \kappa_{\text{ref}} \max \left(\min \left( 1, 1-||a_b - g|| / g \right), 0\right)
$$

where $g$ is the acceleration of gravity and $\kappa_{\text{ref}}$ is chosen as $\kappa_{\text{ref}} = 0.1$. This process is effective to de-drift pitch and roll, however, error accumulation on yaw is unavoidable without fusion using exteroceptive information such as vision.

## 3. 2nd Stage: Kalman Filter

The second stage of the state estimation uses the orientation estimate $\hat{R}_b^o$ along with kinematic measurements from the legs to estimate the base position and velocity. In contrast to previous state estimation techniques that posed this problem as an Extended Kalman Filter (EKF) [2], the two-stage approach allows this secondary fusion to be posed as a conventional Kalman Filter. This simplifies analysis and tuning of the filter and guarantees that the filter equations will never diverge in finite time.

### a. Process Model
In this stage, the state we choose is $x_k = [p_{b,k}^o, v_{b,k}^o, p_{i,k}]^\text{T} \in \mathbb{R}^{18}$, where $p_{b,k}^o$ is the position of the body in world frame at step $k$, $v_{b,k}^o$ is the velocity of the body in world frame at step $k$, and $p_{i,k} \in \mathbb{R}^3$ is the position of foot $i$ at step $k$.

In discrete time, the process equations is modeled as

$$
\begin{aligned}
    &p_{b,k}^o = p_{b,k-1}^o + v_{b,k-1}^o \Delta t + \frac{1}{2}\left(\hat{R}_{b,k}^oa_{b,k}^b + a_g^o\right) \Delta t^2 \\
    &v_{b,k}^o = v_{b,k-1}^o + \left(\hat{R}_{b,k}^oa_{b,k}^b + a_g^o\right) \Delta t \\
    &p_{i,k} = p_{i,k-1}, \quad \forall i = \{1, 2, 3, 4\}
\end{aligned}
$$

where $a_g = [0, 0, -g]^\text{T}$ is the gravitational acceleration, $a_{b,k-1}^o = \hat{R}_{b,k}^oa_{b,k}^b + a_g^o$ is the acceleration of the body in world frame.
In matrix form, we can write

$$
\begin{bmatrix}
p_{b,k}^o \\ v_{b,k}^o \\ p_{i,k}
\end{bmatrix} = 
\begin{bmatrix}
\mathbf{I}_3 & \mathbf{I}_3 \Delta t & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 \\
\mathbf{0}_{12} & \mathbf{0}_{12} & \mathbf{I}_{12}
\end{bmatrix}
\begin{bmatrix}
p_{b,k-1}^o \\ v_{b,k-1}^o \\ p_{i,k-1}
\end{bmatrix}
+ \begin{bmatrix}
\frac{1}{2}\mathbf{I}_3\Delta t^2 \\ \mathbf{I}_3 \Delta t \\ \mathbf{0}_{12\times3}
\end{bmatrix}
a_{b,k-1}^o
$$

This equation leads to the $x_k = Ax_{k-1} + Bu_{k-1} + w_{k-1}$ in Kalman Filter, where $w_{k} \sim (0, Q_k)$ is the process noise term. In simulation, we will get the true robot state if we set $Q_k$ as zero matrix. This is different in real experiment.

### b. Measurement Model
**Leg kinematics** provide measurements of the relative position vector between each foot and the body to dedrift estimates $\hat{p}_b^o$, $\hat{v}_b^o$ and $\hat{p}_i^o$ for each foot.

Letting $p_{\text{rel}}^o(q_i, \hat{R}_{b}^o)$ denote the relative foot position as computed by kinematics, a measurement residual is generated

$$
e_{p,i} = (\hat{p}_i^o - \hat{p}_b^o) - p_{\text{rel}}^o(q_i, \hat{R}_{b}^o)
$$

Similarly, the velocity of the foot relative to the body can be computed from the leg angles, velocities, and the body orientation and angular velocity. This computation is denoted as $\dot{p}_{\text{rel}}^o(q_i, \dot{q}_i, \hat{R}_{b}^o, \omega_b^b)$. *Under the assumption that each foot is fixed (this means that $\hat{v}_i^o = 0$ for $\forall i = 1,2,3,4$)*, this provides an associated measurement residual

$$
\begin{aligned}
e_{v,i} &= (\hat{v}_i^o-\hat{v}_b^o) - \dot{p}_{\text{rel}}^o(q_i, \dot{q}_i, \hat{R}_{b}^o, \omega_b^b) \\
&= (-\hat{v}_b^o) - \dot{p}_{\text{rel}}^o(q_i, \dot{q}_i, \hat{R}_{b}^o, \omega_b^b)
\end{aligned}
$$

Finally, a contact height $h_i$ is assumed for each foot with associated measurement residual:

$$
e_{h_i} = (\begin{bmatrix}
    0 & 0 & 1
\end{bmatrix}\hat{p}_i^o) - h_i
$$

Using the measurement residual equations described above, we can build our measurement model, which is

$$
\begin{aligned}
    &p_{\text{rel},k}^o(q_{i,k}, \hat{R}_{b,k}^o) = (\hat{p}_{i,k}^o - \hat{p}_{b,k}^o) \\
    &\dot{p}_{\text{rel},k}^o(q_{i,k}, \dot{q}_{i,k}, \hat{R}_{b,k}^o, \omega_{b,k}^b) = \hat{v}_{b,k}^o \\
    &h_{i,k} = (\begin{bmatrix}
    0 & 0 & 1    
    \end{bmatrix}\hat{p}_{i,k}^o)
\end{aligned}
$$