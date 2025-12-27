"""
Local Flask proxy that serves the frontend AND proxies to Big Red
This bypasses CORS by serving everything from the same origin
"""
from flask import Flask, request, jsonify, send_file
import requests
import os

app = Flask(__name__)
BIG_RED_URL = "http://localhost:5000"  # SSH tunnel to Big Red

@app.route('/')
def index():
    return send_file('/Users/madhavanbalaji/Documents/OmniBridge/frontend/blip2_demo.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Proxy to Big Red BLIP-2 server"""
    try:
        file = request.files.get('file')
        files = {'file': (file.filename, file.stream, file.content_type)}
        response = requests.post(f"{BIG_RED_URL}/generate", files=files, timeout=120)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  OmniBridge Local Proxy")
    print("  Open: http://localhost:3000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=3000, debug=False)
