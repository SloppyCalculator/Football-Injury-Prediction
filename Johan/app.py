from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from Flask backend!"})

@app.route('/sign')
def sign():
    return render_template('sign.html')  # This will render the sign.html from templates

if __name__ == '__main__':
    app.run(debug=True)


