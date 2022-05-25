from flask import Flask, render_template, jsonify, request
from time import time
import subprocess

app = Flask(__name__, template_folder = "/root/proyek-pm/templates")

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods = ['POST'])
def upload():
    file = request.files['file']
    now = int(time())
    filename = f"{request.form['email']}-{now}.csv"
    file.save(f'uploads/{filename}')
    bashCommand = f"python3 test.py {filename} {request.form['email']}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output, error)
    return jsonify({'ok' : 'ok'})

if __name__ == "__main__":
    app.run(debug=True)

