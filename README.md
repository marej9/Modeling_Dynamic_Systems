# Modeling Dynamical Systems  
Research Project at the Chair of Information-Oriented Control, Technical University of Munich (TUM)

## Project Overview
This project focuses on the modeling and comparison of different approaches for time series prediction of dynamical systems.  
The goal is to analyze machine learning models and system-theoretic methods across synthetic and real-world datasets.

---

## Objective
- Analyze which model is best suited for different types of dynamical systems  
- Compare models based on:
  - Prediction accuracy  
  - Training time  
  - Model complexity  

---

## Models Implemented

### Recurrent Neural Networks (RNN)
- Captures temporal dependencies using hidden states  
- Efficient for sequential data  

### Transformer (Encoder-based)
- Uses attention mechanisms to model relationships in sequences  
- Strong performance on complex and noisy data  

### Koopman Operator
- Transforms nonlinear dynamics into a linear representation  
- Enables analysis using linear system theory  

---

## Datasets

- **Van der Pol Oscillator** (deterministic nonlinear system)  
- **Van der Pol Oscillator (with noise)**  
- **Lorenz System** (chaotic system)  
- **Real-world temperature data (IoT sensors)**  

---

## Methodology

- Data standardization before training  
- Train / validation / test split  
- Evaluation based on:
  - Test loss  
  - Training time  
  - Implementation complexity  

---

## Key Results

### RNN
- Most efficient for deterministic systems  
- Fast training with competitive accuracy  
- Performance decreases with noisy data  

### Transformer
- Best performance on noisy datasets  
- High computational cost  
- Less stable for long-term predictions  

### Koopman Model
- Strong performance on structured real-world data  
- Less effective for highly nonlinear or chaotic systems  
- Training time increases with sequence length  

---

## Conclusion

- No single model performs best in all scenarios  
- Model selection depends on:
  - System characteristics  
  - Noise level  
  - Computational resources  

---

## Tech Stack

- Python  
- PyTorch  
- Koopman framework  

---

## My Contribution

- Implementation of models  
- Data preprocessing and pipeline design  
- Training and evaluation of different architectures  
- Comparative analysis of model performance  

---

## Additional Resources

- Report: [https://github.com/marej9/Modeling_Dynamic_Systems/blob/main/How_to_Model_Dynamical_Systems_report.pdf]  
- Presentation: [https://github.com/marej9/Modeling_Dynamic_Systems/blob/main/How%20to%20Model%20Dynamical%20Systems_presentation.pdf]  

---
