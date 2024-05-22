from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Định nghĩa cấu trúc của mạng Bayes
model = BayesianNetwork([
    ('Smoking', 'LungCancer'),
    ('Genetics', 'LungCancer'),
    ('LungCancer', 'Cough'),
    ('LungCancer', 'ChestPain')
])

# Xác định phân phối xác suất có điều kiện (CPD)
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

# Thêm CPDs vào model
model.add_cpds(cpd_smoking, cpd_genetics, cpd_lung_cancer, cpd_cough, cpd_chest_pain)

# Kiểm tra xem model có hợp lệ không?
assert model.check_model()

# Thực hiện suy luận
infer = VariableElimination(model)

# Truy vấn xác suất ung thư phổi đưa ra bằng chứng
evidence = {'Smoking': 1, 'Genetics': 0}  # Dẫn chứng: nguoi hut thuoc la va khong co khuynh huong di truyen
result = infer.query(variables=['LungCancer'], evidence=evidence)
print(result)

# Truy vấn xác suất bị ho do ung thư phổi
evidence = {'LungCancer': 1}
result = infer.query(variables=['Cough'], evidence=evidence)
print(result)
