from flask import Flask, render_template, request
from search import score, retrieve, build_index, initialize_models
from time import time
import os

app = Flask(__name__, template_folder='.')

if __name__ == '__main__': 
	build_index(os.path.dirname(os.path.realpath(__file__)))
	initialize_models()
	print('done')
	
@app.route('/', methods=['GET'])
def index():
	start_time = time()
	query = request.args.get('query')
	if query is None:
		query = ''
		
	if __name__ == '__main__': 
		documents = retrieve(query)
		documents = sorted(documents, key=lambda doc: -score(query, doc))[:50]
	else:
		documents = []
		
	results = [doc.format(query)+['%.2f' % score(query, doc)] for doc in documents] 
	return render_template(
	'index.html',
	time="%.2f" % (time()-start_time),
	query=query,
	search_engine_name='Search',
	results=results
)

if __name__ == '__main__':
	app.run(debug=False)
