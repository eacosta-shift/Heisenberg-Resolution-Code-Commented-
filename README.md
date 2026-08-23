# Heisenberg Limited Resolution of Phase Estimation

This repository contains the following:

1. No Seeding

   - bounds.py : Computes the Quantum and Classical Fisher Information, as well the inverse of the propagation of errors. This quantities are plotted as a function of φ.

   - losses.py : Computes the Quantum and Classical Fisher Information, as well the inverse of the propagation of errors for different values of τ. This quantities are plotted as a function of φ and with τ as a parameter. Figures 4.1 and 4.2 of the thesis manuscript were generated with this script.

2. Seeding

   - N_s2.py : Computes the average number of output signal photons for the cases of α_i=0 and α_i=x, where x may be any arbitrary value. Then, it plots both quantities as a function of φ for comparisson.

   - bounds.py : Computes the Quantum Fisher Information and the inverse of the propagation of errors when considering seeding. This quantities are plotted as a function of φ. The Classical Fisher Information is not presented because there is not analytical formula.

   - building_blocks.ipynb : Computes and plots multiple quantities involved in the eventual computation of the Classical Fisher Information, such as the probability dist. p_m and its derivative with respect to φ, d/dφ(p_m). This allows to observe how each of these quantities behave when varying the different parameters.

   - summands.ipynb : For a single combination of parameters φ, G, τ and α, calculates and displays the summands required to compute the Classical Fisher Information up to a given $m_{Max}$.

   - npFisher.py : Uses NUMPY to evaluate the expression of the classical Fisher information in the case of seeding. The first plot presents the summands (up to a $m_{Max}$) that are added to obtain the classical Fisher info $F_c(φ=0)$. Since the number of summands required for the sum to converge increases as φ goes to zero, the value of $m_{Max}$ that results in a convergent sum at φ=0 would IN PRINCIPLE lead to convergence for all values of φ.  The second plot displays the Classical Fisher Info as a function of φ, compared to the inverse of the propagation of errors. Comments on the cause of numerical limitations can be found directly on this .py file.

   - mpFisher.py : Uses MPMATH (Python library for real and complex floating-point arithmetic with arbitrary precision) to evaluate the expression of the classical Fisher information in the case of seeding. This allows to overcome the numerical limitations of numpy, at the price of greatly decreased computational speed. This is the valid and best way I implemented to compute the classical fisher information in the case of seeding, and it shows that, when there's seeding, there's is a (slight) difference between the classical fisher info and the inverse of the propagation of errors. Yhis result contradicts what was reported on the thesis manuscript, as there was no time to adecquately explore this before submission.  Moving forward, classical Fisher Info and Inverse prop. of errors are treated as synonymous due to high computational cost.

3. Finding φ_opt

   - OptimalPhase.py : This file contains a function to find the optimal phase given a set of values for the parameter space, as defined by eqs. 4.10 and 4.11 of the thesis manuscript. Numpy once again runs into numerical limitations when trying to compute $φ_{opt}$ for high values of the parametric gain G, which is why it is necessary to use the mpmath library again. The function is defined in such a way that it is possible to input and output numpy vectors, even if behind scenes operations are done using mpmath to achieve the required numerical precision.

   - OptPhase_Widget.ipynb : This notebook contains a dynamical widget that, for a given set of values of the parameter space, displays the classical Fisher information (inv. prop of errors) as a function of φ, as well as the optimal phase $φ_{opt}$ that maximizes this quantity. The parameter space can be defined via sliders.

4. Fisher Info v. Gain

    -  figures. : Contains the code that was used to generate the plots in figures 4.4, 4.5 and 4.9 of the thesis manuscript.

    - seed.py : Contains the code that was used to generate the plots in figures 4.6, 4.7 and 4.8 of the thesis manuscript, as well as plots that were presented only in the defense.

    - FvG_Widget.py : This notebook contains a dynamical widget that, for a given set of values of the parameter space, displays the optimal classical Fisher information (inv. prop of errors) $F(\varphi_{opt})$ as a function of G, as well as the same quantity normalized by the number of down-converted photons. The values of parameter space can be defined via sliders.
   
5. Coincidences

   -> numpy
   - characterization.ipynb : Contains a detailed explanation on how the probability distribution $p_{mn}$. This is a complex expression, and requires several python functions to compute. Here, these functions are introduced for the first time, and this should be reviewed even if using the mpmath version, because the python functions used are introduced here for the first time.

   - pmn_physical.py : Computes the $p_{mn}$ probability distribution up to a given $(n_{max}, m_{max})$ For a single combination of parameters φ, G, τ and α. However, it can be observed that even though the script works for low values of $(m,n)$, there are numerical imprecissions that make this function unusable when trying to compute $p_{mn}$ for higher values of $(m,n)$

   - pmn_validation_np.py : Compares the numerical algortihm developed in pmn_physical.py to the theoretical expressions of $p_{mn}$ for low values $(m,n)$. These expressions can be found in Prof. Torres' notes

   - Fisher_Info_np.py : Uses the python function to get the probability distribution $p_{mn}$ to get the Fisher Information in the case of coincidences.

   -> mpmath
   - pmn_pmmath.py : In order to tackle the numerical inaccuracies when using numpy, the probability distribution is computed by using the mpmath library. Relevant information to understand this pyhton function can be found in the numpy/characterization.ipynb file. 

   - pmn_validation_mp.py : Compares the numerical algortihm developed in pmn_pmmath.py to the theoretical expressions of $p_{mn}$ for low values $(m,n)$. These expressions can be found in Prof. Torres' notes

   - Fisher_Info_mp.py : Uses the python function (mpmath) to get the probability distribution $p_{mn}$ to get the Fisher Information in the case of coincidences. This is the version with no numerical errors, and what was used to create the figures in the thesis manuscript. 

6. Singles v. Coincidences

   - Info v. Gain : Some not used plots of Fisher information (singles and coincidences) as a function of gain.

   - Info v. Losses : Generates plots of Fisher Information (singles and coincidences) as a function $|\tau|^2$ 


singles.py : Self-contained python file with all the functions necessary to compute the fisher information in the case of singles, as well as the optimal phase $\varphi_{opt}$. It sometimes imported by some of the other files, so it is imperative to rely on this file to run some of the others. Since this file contains all the relevant functions, it can be called in other python files to do further coding.

coincidences.py : Self-contained python file with all the functions necessary to compute the fisher information in the case of coincidences. It sometimes imported by some of the other files, so it is imperative to rely on this file to run some of the others. Since this file contains all the relevant functions, it can be called in other python files to do further coding. 

.venv : contains the necessary packages and package versions to run a the code. Particularly important to run the dynamical widgets.

   
