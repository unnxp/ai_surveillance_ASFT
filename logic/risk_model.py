import torch
import torch.nn as nn

class RiskMLP(nn.Module):
    """
    โครงข่ายประสาทเทียมขนาดเล็ก (Multi-Layer Perceptron) 
    สำหรับจำแนกและประเมินระดับความเสี่ยง (LOW, MEDIUM, HIGH, CRITICAL)
    จากคุณสมบัติทางพฤติกรรม 6 มิติ
    """
    def __init__(self, input_dim=6, num_classes=4):
        super(RiskMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
