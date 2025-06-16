
module spider scipy-stack/2020a #to get scipy version numbers

module load python/3.11
module load kahypar
module load scipy-stack/??
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index kahypar numpy qiskit qiskit-aer networkx matplotlib pylatexenc pytket pytket-qiskit hypernetx celluloid igraph decorator pygraphviz importlib_resources pyzx pytket-pyzx
pip download --no-deps git+https://github.com/CQCL/pytket-dqc.git
pip install --no-index path/to/pytket-dqc
pip download --no-deps git+https://github.com/felix-burt/DISQCO.git
pip install --no-index path/to/DISQCO

pip freeze --local > $HOME/requirements.txt
deactivate
rm -rf $ENVDIR


# copy over files: job should be in home, so are python scripts
# ccdb_job.sh
# main.py only, but modify n_qubits