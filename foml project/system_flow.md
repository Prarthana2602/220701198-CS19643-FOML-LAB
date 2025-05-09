```mermaid
graph TD
    A[User] -->|Access| B[Web Interface]
    B -->|Submit Form| C[Flask Application]
    C -->|Load| D[ML Model]
    D -->|Load| E[model.joblib]
    
    subgraph "Input Processing"
        C -->|Process| F[Feature Extraction]
        F -->|Convert| G[Salary Mapping]
        G -->|Create| H[Feature Array]
    end
    
    subgraph "Prediction"
        H -->|Predict| I[Model Prediction]
        I -->|Calculate| J[Probability Calculation]
        J -->|Format| K[Response]
    end
    
    K -->|Return| B
    B -->|Display| A

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbb,stroke:#333,stroke-width:2px
    style E fill:#fbb,stroke:#333,stroke-width:2px
```

## System Flow Description

1. **User Interface**
   - Users access the web interface through their browser
   - Interface collects employee data through a form

2. **Flask Application**
   - Handles HTTP requests and responses
   - Manages the prediction workflow
   - Loads the pre-trained ML model

3. **Input Processing**
   - Extracts features from form data
   - Maps salary categories (low/medium/high) to numerical values
   - Creates feature array for model input

4. **Prediction Pipeline**
   - Model makes prediction using input features
   - Calculates probability of employee staying/leaving
   - Formats response with probabilities

5. **Response**
   - Returns JSON response with prediction results
   - Displays results to user through web interface

## Technical Components

- **Frontend**: HTML/CSS/JavaScript (templates)
- **Backend**: Flask (Python)
- **ML Model**: Pre-trained model (model.joblib)
- **Data Processing**: NumPy for array operations
- **Model Persistence**: Joblib for model storage 