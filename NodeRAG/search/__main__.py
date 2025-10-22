import argparse
import yaml
import os
import sys # Import sys for exiting gracefully
from flask import Flask, request, jsonify

from .search import NodeSearch
from ..config import NodeConfig

parser = argparse.ArgumentParser(description='TGRAG search engine')
parser.add_argument('-f', '--folder_path', type=str, required=True, help='The folder path containing the Node_config.yaml file')
args = parser.parse_args()

# 1. More robust path handling and error checking
if not os.path.isdir(args.folder_path):
    print(f"Error: The provided folder path does not exist: {args.folder_path}")
    sys.exit(1)

config_path = os.path.join(args.folder_path, 'Node_config.yaml')

if not os.path.exists(config_path):
    print(f"Error: 'Node_config.yaml' not found in the specified folder: {args.folder_path}")
    sys.exit(1)

# 2. Load config and instantiate NodeConfig once
try:
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # This single instance will be used for the search engine
    node_config = NodeConfig(config_data)

except Exception as e:
    print(f"Error loading or parsing the configuration file: {e}")
    sys.exit(1)

# 3. Provide clear feedback on which backend is being used
# This reads the 'graph_db_type' that NodeConfig loaded from the YAML file.
active_backend = getattr(node_config, 'graph_db_type', 'networkx')
print("="*50)
print(f"✅ Search engine initializing with '{active_backend}' backend...")
print("="*50)


# Initialize the search engine with the config object
# The NodeSearch class will now handle the backend choice internally.
search_engine = NodeSearch(node_config)

app = Flask(__name__)

url = node_config.config.get('url', '127.0.0.1')
port = node_config.config.get('port', 5000)


@app.route('/answer', methods=['POST'])
def answer():
    question = request.json['question']
    answer = search_engine.answer(question)
    return jsonify({'answer': answer.response})

@app.route('/answer_retrieval', methods=['POST'])
def answer_retrieval():
    question = request.json['question']
    answer = search_engine.answer(question)
    return jsonify({'answer': answer.response, 'retrieval': answer.retrieval_info})

@app.route('/retrieval', methods=['POST'])
def search():
    question = request.json['question']
    retrieval = search_engine.search(question)
    return jsonify({'retrieval': retrieval.retrieval_info})

if __name__ == '__main__':
    print(f"🚀 Flask server starting at http://{url}:{port}")
    app.run(host=url, port=port, debug=False, threaded=True)