from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "defense-automation"})

@app.route('/')
def root():
    return jsonify({"service": "defense-automation", "phase": 8})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
