data_mode='dev' # Options: 'dev', 'train' 

data_path="./data/dev/dev.json" # UPDATE THIS WITH THE PATH TO THE TARGET DATASET

pipeline_nodes='keyword_extraction+entity_retrieval+context_retrieval+column_filtering+table_selection+column_selection+candidate_generation+revision+evaluation'
checkpoint_nodes=''
checkpoint_dir=""

# Nodes:
    # keyword_extraction
    # entity_retrieval
    # context_retrieval
    # column_filtering
    # table_selection
    # column_selection
    # candidate_generation
    # revision
    # evaluation


entity_retieval_mode='ask_model' # Options: 'corrects', 'ask_model'

context_retrieval_mode='vector_db' # Options: 'corrects', 'vector_db'
top_k=5

table_selection_mode='ask_model' # Options: 'corrects', 'ask_model'

column_selection_mode='ask_model' # Options: 'corrects', 'ask_model'

engine1='gemini-pro'
engine2='gpt-3.5-turbo-0125'
engine3='gpt-4-turbo'
engine4='claude-3-opus-20240229'
engine5='gemini-1.5-pro-latest'
engine6='finetuned_nl2sql'
engine7='meta-llama/Meta-Llama-3-70B-Instruct'
engine8='finetuned_colsel'
engine9='finetuned_col_filter'
engine10='gpt-3.5-turbo-instruct'

pipeline_setup='{
    "keyword_extraction": {
        "engine": "'${engine2}'",
        "temperature": 0.2,
        "base_uri": ""
    },
    "entity_retrieval": {
        "mode": "'${entity_retieval_mode}'"
    },
    "context_retrieval": {
        "mode": "'${context_retrieval_mode}'",
        "top_k": '${top_k}'
    },
    "column_filtering": {
        "engine": "'${engine2}'",
        "temperature": 0.0,
        "base_uri": ""
    },
    "table_selection": {
        "mode": "'${table_selection_mode}'",
        "engine": "'${engine3}'",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    },
    "column_selection": {
        "mode": "'${column_selection_mode}'",
        "engine": "'${engine3}'",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    },
    "candidate_generation": {
        "engine": "'${engine3}'",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    },
    "revision": {
        "engine": "'${engine3}'",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    }
}'

echo -e "${run_name}"
python3 -u ./src/main.py --data_mode ${data_mode} --data_path ${data_path}\
        --pipeline_nodes ${pipeline_nodes} --pipeline_setup "$pipeline_setup"\
        # --use_checkpoint --checkpoint_nodes ${checkpoint_nodes} --checkpoint_dir ${checkpoint_dir}
  
