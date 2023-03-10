# Project 1 - Riskfuel 


![](https://media-exp1.licdn.com/dms/image/C4D0BAQHa_yrMUj4Fwg/company-logo_200_200/0/1575957122798?e=2159024400&v=beta&t=Te0m8CUYKG3PNIkwZd4rWo1ZwQm0_lAB60hHWA-S6po)


## Introduction

This project consists of training a deep neural network to approximate the analytic Black-Scholes price of a European Put option. The neural net will be evaluated on a validation set defined on a closed domain of input parameters. Your goal is to obtain the best model performance (ie. Mean and Maximum Squared Error). 


## Black-Scholes Model 

The Black-Scholes Model was concieved by McMaster alumni Myron Scholes and his co-author Fischer Black. From the Black-Scholes model, one can derive an analytic solution to a standard European Put option. The analytic solution has been provided to you in `utils/black_scholes.py`. The pricer takes in the following variables: 

- Underlying stock price (S) 
- Strike price (K)
- Time to maturity (T)
- Risk free interest rate (r) 
- Volatility (sigma)

Which then outputs the following price: 

- Price of put option in dollars (value) 

Essentially it is a function which takes in 5 inputs, and returns 1 output. 


![](media/bsm.png)


## Domain
The domain on which the model will be validated on. 

```yaml
S_bound = [0.0, 200.0]
K_bound = [50.0, 150.0] 
T_bound = [0.0, 5.0]
r_bound = [0.001, 0.05]
sigma_bound = [0.05, 1.5]

```

There is a file called `dataset.py` that will assist you in generating a training/validation set. 

## Constraints/Requirements 

The `riskfuel_test.py` file has validation code describing how each person will be evaluated. Both myself and the TA must be able to download your git repo, run `pip install -r requirements.txt`, and then run 

```bash 
python riskfuel_test.py --data_frame_name <validation.csv>
```   

The output should look something like this: 

```bash 
 ============ Evaluating Student: Nik Pocuca ==================== 
 Full Name:
 Nikola Pocuca
 Student ID:
 pocucani
 ================================================================ 

 MODEL PERFORMANCE: 1228.5853271484375 MAX LOSS: 107.51322174072266 

```

The validation code will be checked manually for everyone. 

## Grading Scheme 

This project will use the following grading scheme.
The max error between your model, and the pricer will be used to assign your grade. 

### <u> Level 1 (B-) </u>
To achieve this level, you must submit valid code. I must be able to run your python scripts, install any dependencies, and 
be able to assess you. Failure to do so will result in a grade of 0.0. The model must display a valid max error exceeding $50 
on the validation set. 

### <u> Level 2 (B) </u>
To achieve this level you must submit valid code, and the model must achieve a valid max error less than $50 but greater 
than $5 on the validation set. 

### <u>  Level 3 (A) </u>
To achieve this level, you must submit valid code, and, the model must achieve a valid max error less than $5 but greater than $0.01 
on the validation set. 

### <u> Level 4 (A+) </u>
To achieve this level, you must submit valid code, and, the model must achieve a valid max error of less than or equal to $0.01 (1 cent). 
This is considered a professional level model. 


### *Do not use the analytic pricer as part of your model. This is considered cheating, you will fail automatically and be subject to academic dishonesty * 

You are free to use any packages/frameworks you like provided that they can be run and installed on a `ubuntu 20.04` docker image found here `https://hub.docker.com/_/ubuntu`. When writing code to evaluate your model, feel free to delete the skeleton code within `riskfuel_test.py`. 

## Recommendations 

Pytorch is a great framework for training ML models that has options for CPU training. Pytorch also allows you to implicity define gradients in an interpretive fashion as opposed to other frameworks such as Tensorflow. 

We whole-heartedly advocate Pytorch for this project `https://pytorch.org/` over other packages. You will NOT be penalized for not using Pytorch if you are more comfortable with other packages. 

Some files have been provided for you to start, they are missing some basic ML techniques but provide an overall basis for how to train BSMs. They are given as 
`demo.py`, and `dataset.py`. 


