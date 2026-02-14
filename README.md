# Bayesian A/B Testing Framework

A comprehensive, interactive web application for Bayesian A/B testing with a clean, minimal black-and-white interface. This tool helps data scientists, analysts, and product managers make data-driven decisions using Bayesian statistics.

![Bayesian A/B Testing Framework](https://via.placeholder.com/800x400?text=Bayesian+A/B+Testing+Framework)


## Overview

This framework implements Bayesian methods for A/B testing, providing intuitive probabilistic interpretations and decision-making tools. Unlike traditional frequentist approaches, Bayesian A/B testing offers:

- **Probabilistic results** — Direct statements like *"95% probability that B is better than A"*
- **Sequential analysis** — Update beliefs as data arrives without "peeking" penalties
- **Decision theory** — Expected loss calculations for business decisions
- **Prior knowledge** — Incorporate historical information into analysis

## Features

### 1. Bayesian Analysis

- Beta-Binomial model for conversion rates  
- Posterior distribution visualization  
- Uplift analysis (absolute and relative)  
- Probability calculations (P(B > A))  

### 2. Sequential Testing

- Real-time posterior updates  
- Probability evolution over time  
- Early stopping recommendations  
- Batch-wise data processing  

### 3. Method Comparison

- Side-by-side Bayesian vs Frequentist results  
- Credible intervals vs Confidence intervals  
- Bayes Factor calculation  
- Practical significance testing  

### 4. Experiment Design

- Sample size calculator  
- Power analysis curves  
- Scenario simulation  
- MDE (Minimum Detectable Effect) planning  

### 5. Educational Resources

- Key concepts explained  
- Interpretation guides  
- Method selection recommendations  
- Interactive learning tools  


## Installation

### Prerequisites

- Python 3.8 or higher  
- pip package manager  

### Step 1: Clone the Repository

```bash
git clone https://github.com/sefabilicier/Bayesian-AB-Testing
cd bayesian-ab-testing
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at:

```
http://localhost:8501
```

## Use Cases

### For Data Scientists

- Validate A/B test results  
- Compare Bayesian vs Frequentist approaches  
- Build intuition about posterior distributions  
- Teach Bayesian concepts to stakeholders  

### For Product Managers

- Make data-driven launch decisions  
- Understand the probability of success  
- Quantify risk and expected loss  
- Determine required sample sizes  

### For Researchers

- Analyze experimental data  
- Incorporate prior knowledge  
- Sequential monitoring of studies  
- Report credible intervals  


## How to Use

### Quick Start Guide

#### Analyze Results

1. Navigate to the **"Analyze"** tab  
2. Choose data input method (Simulated / Manual / CSV)  
3. Set group parameters  
4. Click **"Run Bayesian Analysis"**

#### Interpret Results

- Check **P(B > A)** probability  
- View posterior distributions  
- Analyze uplift distributions  
- Review decision metrics  

#### Explore Sequential Analysis

1. Go to the **"Sequential"** tab  
2. Set number of batches  
3. Watch probability evolve over time  
4. See when you'd reach significance  

#### Compare Methods

1. Visit the **"Compare"** tab  
2. See Bayesian vs Frequentist side-by-side  
3. Understand differences in interpretation  

#### Design Experiments

1. Use the **"Design"** tab for sample size planning  
2. Run scenario simulations  
3. Optimize test parameters  

---

## Key Concepts Explained

### Bayesian A/B Testing

The Beta-Binomial model is used because it is the conjugate prior for binomial data:

- **Prior**: `Beta(α, β)` — represents beliefs before seeing data  
- **Likelihood**: `Binomial(n, p)` — probability of observed conversions  
- **Posterior**: `Beta(α + successes, β + failures)` — updated beliefs  

### Decision Metrics

- **P(B > A)** — Probability treatment outperforms control  
- **Expected Uplift** — Average expected improvement  
- **Expected Loss** — Cost of choosing the wrong variant  
- **Bayes Factor** — Evidence strength for H₁ vs H₀  

---

## 📁 Project Structure

```
bayesian-ab-testing/
│
├── app.py                 # Main Streamlit application
├── bayesian_models.py     # Bayesian and Frequentist models
├── visualizations.py      # Plotting functions
├── utils.py               # Helper utilities
├── requirements.txt       # Dependencies
│
└── README.md              # Project documentation
```

## Example Workflow

### Scenario: Testing a New Website Design

- **Current (Control)**: 10% conversion rate, 1000 visitors  
- **New Design (Treatment)**: 12% conversion rate, 1000 visitors  

### Bayesian Results

- **P(B > A)** = 94.3%  
- **Expected Uplift** = 18.5%  
- **95% Credible Interval** = [2.1%, 35.2%]  
- **Expected Loss (choose B)** = 0.0032  

**Decision:** Moderate evidence to launch the new design, with a 94.3% probability that it outperforms the control.
