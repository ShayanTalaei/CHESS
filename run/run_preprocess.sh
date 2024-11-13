# Define variables
source .env
db_root_directory=$DB_ROOT_DIRECTORY # UPDATE THIS WITH THE PATH TO THE PARENT DIRECTORY OF THE DATABASES
db_id="all" # Options: all or a specific db_id
verbose=true
signature_size=100
n_gram=3
threshold=0.01

# Run the Python script with the defined variables
python3 -u ./src/preprocess.py --db_root_directory "${db_root_directory}" \
                              --signature_size "${signature_size}" \
                              --n_gram "${n_gram}" \
                              --threshold "${threshold}" \
                              --db_id "${db_id}" \
                              --verbose "${verbose}"
