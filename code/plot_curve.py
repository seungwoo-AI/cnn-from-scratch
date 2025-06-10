import pickle, numpy as np
import matplotlib.pyplot as plt

# 1) 로그 로드
loss_record = pickle.load(open("loss_record.pkl", "rb"))

# 2) 평균 곡선 계산 (S-Adam 예시)
model_key = "S-Adam"
loss_arr  = np.array(loss_record[model_key])       # shape (10, 300)
mean_loss = loss_arr.mean(axis=0)
std_loss  = loss_arr.std(axis=0)

# 3) 그리기
epochs = np.arange(1, len(mean_loss)+1)
plt.plot(epochs, mean_loss, label=f"{model_key} mean")
plt.fill_between(epochs, mean_loss-std_loss, mean_loss+std_loss,
                 alpha=0.3, label="±1 std")

plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("S-Adam Loss Curve")
plt.legend(); plt.tight_layout()
plt.savefig("fig1_loss.png", dpi=300)
