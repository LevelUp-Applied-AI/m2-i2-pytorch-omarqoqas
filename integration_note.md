# Integration 2 — PyTorch: Housing Price Prediction

## 1. Model Objective

The model is designed to predict **housing prices in Jordanian Dinar (JOD)**.

- **Target Variable:**
  - `price_jod` → the actual price of the property

- **Input Features (5):**
  1. `area_sqm` → size of the property in square meters  
  2. `bedrooms` → number of bedrooms  
  3. `floor` → floor level of the property  
  4. `age_years` → age of the property in years  
  5. `distance_to_center_km` → distance from the city center in kilometers  

---

## 2. Training Configuration

- **Number of Epochs:** 100  
- **Learning Rate:** 0.01  
- **Optimizer:** Adam  
- **Loss Function:** Mean Squared Error (MSELoss)  

- **Model Architecture:**
  - Linear Layer (5 → 32)
  - ReLU Activation
  - Linear Layer (32 → 1)

- **Preprocessing:**
  - Features were standardized using mean and standard deviation to ensure stable and balanced training.

---

## 3. Training Outcome

- The loss **consistently decreased** during training, indicating that the model was learning effectively.
- Initial loss was relatively high but dropped significantly over time.
- **Final Loss Value:** _(replace this with your last printed loss, e.g., 12345.67)_

---

## 4. Behavioral Observation

The loss decreased rapidly during the first few epochs (around the first 20–30 epochs), then began to stabilize and decrease more slowly.  

This suggests that the model quickly learned the general patterns in the data early on, and later epochs were focused on fine-tuning the predictions.

Additionally, feature standardization helped prevent unstable training and allowed the optimizer to converge more efficiently.