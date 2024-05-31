# Import các thành phần cần thiết từ thư viện pgmpy_ một thư viện Python mạnh mẽ được sử dụng để làm việc với các mô hình đồ thị xác suất, cụ thể là mạng Bayesian
from pgmpy.models import BayesianNetwork
# BayesianNetwork là lớp dùng để tạo và làm việc với các mạng Bayesian, biểu thị các biến ngẫu nhiên và mối quan hệ nhân quả giữa chúng.
from pgmpy.factors.discrete import TabularCPD
# TabularCPD là lớp dùng để định nghĩa các Phân phối Xác suất Có Điều kiện (Phân phối Xác suất Có Điều kiện.) dưới dạng bảng
# xác định xác suất của một biến dựa trên các biến khác.
from pgmpy.inference import VariableElimination
# VariableElimination là lớp cung cấp các phương pháp để thực hiện suy luận trong mạng Bayesian, sử dụng thuật toán loại bỏ biến.


# Định nghĩa cấu trúc của Mạng Bayesian
# Tạo một mạng Bayesian với các mối quan hệ có hướng giữa các biến
model = BayesianNetwork([
    ('Smoking', 'LungCancer'),   # Hút thuốc ảnh hưởng đến ung thư phổi
    ('Genetics', 'LungCancer'),  # Yếu tố di truyền ảnh hưởng đến ung thư phổi
    ('LungCancer', 'Cough'),     # Ung thư phổi ảnh hưởng đến ho
    ('LungCancer', 'ChestPain')  # Ung thư phổi ảnh hưởng đến đau ngực
])

# Định nghĩa các Phân phối Xác suất có Điều kiện (CPDs) cho từng biến
# CPD cho biến Smoking (Hút thuốc)
cpd_smoking = TabularCPD(variable='Smoking', variable_card=2, values=[[0.3], [0.7]])
# P(Smoking=Yes) = 0.3, P(Smoking=No) = 0.7

# CPD cho biến Genetics (Yếu tố di truyền)
cpd_genetics = TabularCPD(variable='Genetics', variable_card=2, values=[[0.1], [0.9]])
# P(Genetics=Yes) = 0.1, P(Genetics=No) = 0.9

# CPD cho biến LungCancer (Ung thư phổi) dựa trên Smoking và Genetics
cpd_lung_cancer = TabularCPD(
    variable='LungCancer', variable_card=2,
    values=[[0.99, 0.90, 0.80, 0.30], [0.01, 0.10, 0.20, 0.70]],
    evidence=['Smoking', 'Genetics'],
    evidence_card=[2, 2]
)

# P(LungCancer | Smoking, Genetics) xác suat bi hoac khong bi ung thu phoi phu thuoc vao 2 yeu to
# P(LungCancer=No | Smoking=Yes, Genetics=Yes) = 0.99
# P(LungCancer=No | Smoking=Yes, Genetics=No) = 0.90
# P(LungCancer=No | Smoking=No, Genetics=Yes) = 0.80
# P(LungCancer=No | Smoking=No, Genetics=No) = 0.30
# P(LungCancer=Yes | Smoking=Yes, Genetics=Yes) = 0.01
# P(LungCancer=Yes | Smoking=Yes, Genetics=No) = 0.10
# P(LungCancer=Yes | Smoking=No, Genetics=Yes) = 0.20
# P(LungCancer=Yes | Smoking=No, Genetics=No) = 0.70

# CPD cho biến Cough (Ho) dựa trên LungCancer
cpd_cough = TabularCPD(
    variable='Cough', variable_card=2,
    values=[[0.6, 0.3], [0.4, 0.7]],
    evidence=['LungCancer'],
    evidence_card=[2]
)
# P(Cough | LungCancer)
# P(Cough=No | LungCancer=No) = 0.6
# P(Cough=No | LungCancer=Yes) = 0.3
# P(Cough=Yes | LungCancer=No) = 0.4
# P(Cough=Yes | LungCancer=Yes) = 0.7

# CPD cho biến ChestPain (Đau ngực) dựa trên LungCancer
cpd_chest_pain = TabularCPD(
    variable='ChestPain', variable_card=2,
    values=[[0.8, 0.4], [0.2, 0.6]],
    evidence=['LungCancer'],
    evidence_card=[2]
)
# P(ChestPain | LungCancer)
# P(ChestPain=No | LungCancer=No) = 0.8
# P(ChestPain=No | LungCancer=Yes) = 0.4
# P(ChestPain=Yes | LungCancer=No) = 0.2
# P(ChestPain=Yes | LungCancer=Yes) = 0.6

# Thêm các CPDs vào mô hình
model.add_cpds(cpd_smoking, cpd_genetics, cpd_lung_cancer, cpd_cough, cpd_chest_pain)

# Kiểm tra xem mô hình có hợp lệ hay không
assert model.check_model()
# Nếu mô hình hợp lệ, không có lỗi gì sẽ được ném ra.

# Thực hiện suy luận
infer = VariableElimination(model)
# Sử dụng thuật toán loại bỏ biến để thực hiện suy luận trên mạng Bayesian

# Truy vấn xác suất của Lung Cancer dựa trên bằng chứng
evidence = {'Smoking': 1, 'Genetics': 0}  # Bằng chứng: Người hút thuốc và không có yếu tố di truyền
result = infer.query(variables=['LungCancer'], evidence=evidence)
print(result)
# In ra xác suất bị ung thư phổi khi biết rằng người đó hút thuốc và không có yếu tố di truyền

# Truy vấn xác suất bị ho khi biết bị ung thư phổi
evidence = {'LungCancer': 1}
result = infer.query(variables=['Cough'], evidence=evidence)
print(result)
# In ra xác suất bị ho khi biết rằng người đó bị ung thư phổi

#Giải thích lại dòng 45: Điều này có nghĩa là nếu một người không hút thuốc và không có yếu tố di truyền, xác suất bị ung thư phổi là 70%.