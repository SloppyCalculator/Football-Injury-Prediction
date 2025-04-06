# ⚽️ Football Injury Prediction App Backend

This is the backend for an AI-powered football injury prediction application built with Flask. It provides workload prediction, performance analysis, match readiness estimation, and integrates a chatbot via Rasa to support player and coach interaction. Supabase is used for backend services such as authentication and data storage.

## 🚀 Features

- RESTful API using Flask  
- Injury prediction using ML models  
- Workload prediction and tracking  
- Performance analysis from training and match data  
- Match readiness prediction  
- Interactive chatbot (Rasa) for user engagement  
- User authentication with hashed passwords (bcrypt)  
- Supabase integration for database and auth  
- Cross-Origin Resource Sharing (CORS) support  
- Environment variable configuration using `.env`  
- HTML rendering with `render_template` (optional UI)

## 📦 Dependencies

The following Python packages are required:

- `flask`  
- `flask-cors`  
- `requests`  
- `pandas`  
- `numpy`  
- `joblib`  
- `python-dotenv`  
- `supabase` or `supabase-py`  
- `bcrypt`  
- `rasa` (installed separately in a dedicated virtual environment)

## Files

To access the app and to run the app proceed through the main folder. The other folders consist of the codes used to train the various models

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Ensure Python Version is ≤ 3.9

Rasa supports only Python 3.7–3.9. To check your Python version:

```bash
python --version
```

If your version is higher, use `pyenv`, `conda`, or install Python 3.9 manually. Example with `pyenv`:

```bash
pyenv install 3.9.13
pyenv local 3.9.13
```

### 3. Install Flask and Project Dependencies

Install the required packages globally or in your current Python environment:

```bash
pip install flask flask-cors requests pandas numpy joblib python-dotenv supabase bcrypt
```

Alternatively, if a `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

### 4. Set Up Rasa in a Separate Environment

To avoid conflicts, install Rasa in its own virtual environment:

```bash
python -m venv rasa_env
source rasa_env/bin/activate      # Windows: rasa_env\Scripts\activate
```

Install Rasa:

```bash
pip install rasa
```

Or install a specific version:

```bash
pip install rasa==3.6.10
```

Verify Rasa installation:

```bash
rasa --version
```

### 5. Environment Variables and Compressed files

Use the provided env to access the database. There contains some models which are too large and had to be compressed to be uploaded. Extract those files.

### 6. Run the Flask App

Run the Flask app directly using:

```bash
python app.py
```

Or using Flask CLI:

```bash
export FLASK_APP=app.py           # Windows: set FLASK_APP=app.py
flask run
```

### 7. Run Rasa Chatbot

Activate your Rasa environment:

```bash
source rasa_env/bin/activate      # Windows: rasa_env\Scripts\activate
```
First train using uploaded rasa files as created model is too large to be included or download the trained model from following link https://drive.google.com/drive/folders/10yiVCCl34WwmKhP-OuhWNjQoFnYGqSFk?usp=drive_link

```bash
rasa train
```

Then start the Rasa server:

```bash
rasa run --enable-api
```

To use the Rasa interactive shell or train models:

```bash
rasa train
rasa shell
```

## 📁 Project Structure

```
your-project/
│
├── app.py                  # Main Flask application
├── rasa/                   # Rasa chatbot project files
├── models/                 # ML models for prediction
├── templates/              # HTML templates
├── static/                 # Static assets (CSS, JS)
├── .env                    # Environment variables
└── README.md               # You're here
```
Models do not contain a specific folder and are present externally in the project folder

## 🧠 How It Works

- Flask app handles user authentication, data ingestion, predictions, and analytics.  
- ML models process player data to predict injury risk, workload stress, performance metrics, and match readiness.  
- Rasa chatbot interacts with players or staff for feedback and tracking.  
- Supabase is used for real-time storage and backend services.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 📬 Contact

Made with ❤️ by 
Nadija (https://github.com/Nadija32)
Hashan (https://github.com/Itzz-Hashan)
Tharushi (https://github.com/TharuNethkini)
Johan (https://github.com/SloppyCalculator)  

Feel free to open issues or submit pull requests for improvements!
