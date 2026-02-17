# Taxi-v3: Q-learning vs DQN

This repo compares **tabular Q-learning** and **Deep Q-Network (DQN)** on **Gymnasium Taxi-v3**.
It includes training curves (moving average) and greedy-policy GIF demos after training.

## Results
### Episode return (moving average)
![return](assets/compare_return.png)

### Illegal actions (moving average)
![illegal](assets/compare_illegal.png)

### Policy demos (greedy after training)
**Q-learning:** ![q](assets/demo_Q-learning.gif)  
**DQN:** ![d](assets/demo_DQN.gif)

## Quickstart
```bash
pip install -r requirements.txt
python run_this.py

