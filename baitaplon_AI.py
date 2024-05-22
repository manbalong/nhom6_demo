from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network
model = BayesianNetwork([
    ('Smoking', 'LungCancer'),
    ('Genetics', 'LungCancer'),
    ('LungCancer', 'Cough'),
    ('LungCancer', 'ChestPain')
])

# Define the Conditional Probability Distributions (CPDs)
cpd_smoking = TabularCPD(variable='Smoking', variable_card=2, values=[[0.3], [0.7]])
cpd_genetics = TabularCPD(variable='Genetics', variable_card=2, values=[[0.1], [0.9]])

cpd_lung_cancer = TabularCPD(
    variable='LungCancer', variable_card=2,
    values=[[0.99, 0.90, 0.80, 0.30], [0.01, 0.10, 0.20, 0.70]],
    evidence=['Smoking', 'Genetics'],
    evidence_card=[2, 2]
)

cpd_cough = TabularCPD(
    variable='Cough', variable_card=2,
    values=[[0.6, 0.3], [0.4, 0.7]],
    evidence=['LungCancer'],
    evidence_card=[2]
)

cpd_chest_pain = TabularCPD(
    variable='ChestPain', variable_card=2,
    values=[[0.8, 0.4], [0.2, 0.6]],
    evidence=['LungCancer'],
    evidence_card=[2]
)

# Add CPDs to the model
model.add_cpds(cpd_smoking, cpd_genetics, cpd_lung_cancer, cpd_cough, cpd_chest_pain)

# Check if the model is valid
assert model.check_model()

# Perform inference
infer = VariableElimination(model)

# Query the probability of Lung Cancer given evidence
evidence = {'Smoking': 1, 'Genetics': 0}  # Evidence: Smoker and No genetic predisposition
result = infer.query(variables=['LungCancer'], evidence=evidence)
print(result)

# Query the probability of having a cough given Lung Cancer
evidence = {'LungCancer': 1}
result = infer.query(variables=['Cough'], evidence=evidence)
print(result)
