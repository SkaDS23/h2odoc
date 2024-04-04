Paramètres Expert, Fonction Main (gen.py)
=============

Paramètres expert a modifier avant le lancement de l'application (modifiable dans la base de données)

Paramètres :

- **base_model** (*str*): 
    Model HF-type name. If use --base_model to preload model, cannot unload in gradio in models tab.

- **load_8bit** (*bool*): 
    Load model in 8-bit using bitsandbytes.
    
- **load_4bit** (*bool*): 
    Load model in 4-bit using bitsandbytes.
    
- **low_bit_mode** (*int*): 
    0: no quantization config, 1: change compute, 2: nf4, 3: double quant, 4: 2 and 3.
    See: `Transformers Documentation <https://huggingface.co/docs/transformers/main_classes/quantization>`_
    If using older bitsandbytes or transformers, 0 is required.
    
- **load_half** (*bool*): 
    Load model in float16 (None means auto, which means True unless t5 based model), otherwise specify bool.

- **prompt_type** (*str*): 
    Type of prompt, usually matched to fine-tuned model or plain for foundational model.

- **prompt_dict** (*str*): 
    If prompt_type=custom, then expects (some) items returned by get_prompt(..., return_dict=True)

- **system_prompt** (*str*): 
    Universal system prompt to use if model supports, like LLaMa2, regardless of prompt_type definition.

    Useful for langchain case to control behavior, or OpenAI and Replicate.

    If None, 'None', or 'auto', then for LLaMa or other models that internally have system_prompt, will use default for each model

    If '', then no system prompt (no empty template given to model either, just no system part added at all)

    If some string not in ['None', 'auto'], then use that as system prompt

    Default is '', no system_prompt, because often it hurts performance/accuracy

- **allow_chat_system_prompt** (*bool*): 
    Whether to use conversation_history to pre-append system prompt.

- **temperature** (*float*): 
    Generation temperature.

- **top_p** (*float*): 
    Generation top_p.

- **top_k** (*int*): 
    Generation top_k.

- **penalty_alpha** (*float*): 
    Penalty_alpha>0 and top_k>1 enables contrastive search (not all models support).

- **num_beams** (*int*): 
    Generation number of beams.

- **repetition_penalty** (*float*): 
    Generation repetition penalty.

- **num_return_sequences** (*int*): 
    Generation number of sequences (1 forced for chat).

- **do_sample** (*bool*): 
    Generation sample. Enable for sampling for given temperature, top_p, top_k, else greedy decoding and then temperature, top_p, top_k not used. [More Info](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.do_sample)

- **seed** (*int*): 
    Seed (0 means random seed, >0 uses that seed for sampling so reproducible even for sampling). None becomes 0.

- **max_new_tokens** (*int*): 
    Generation max new tokens.

- **min_new_tokens** (*int*): 
    Generation min tokens.

- **early_stopping** (*bool*): 
    Generation early stopping.

- **max_time** (*float*): 
    Maximum time to allow for generation.

- **memory_restriction_level** (*int*): 
    0 = no restriction to tokens or model, 1 = some restrictions on token 2 = HF like restriction 3 = very low memory case.

- **debug** (*bool*): 
    Enable debug mode.

- **save_dir** (*str*): 
    Directory chat data is saved to.

- **max_seq_len** (*int*): 
    Manually set maximum sequence length for the LLM.

- **max_output_seq_len** (*int*): 
    Manually set maximum output length for the LLM.

- **src_lang** (*str or None*): 
    Source languages to include if doing translation (None = all).

- **tgt_lang** (*str or None*): 
    Target languages to include if doing translation (None = all).

- **close_button** (*bool*): 
    Whether to show close button in system tab (if not public).

- **chat** (*bool*): 
    Whether to enable chat mode with chat history.

- **chat_conversation** (*list of tuples*): 
    List of tuples of (human, bot) conversation pre-appended to existing chat when using instruct/chat models. Requires also `add_chat_history_to_context = True`. It does *not* require `chat=True`, so works with nochat_api etc.

- **stream_output** (*bool*): 
    Whether to stream output.

- **async_output** (*bool*): 
    Whether to do asyncio handling.

- **num_async** (*int*): 
    Number of simultaneously allowed asyncio calls to make for async_output. Too many will overload the inference server, too few will be too slow.

- **login_mode_if_model0** (*bool*): 
    Set to True to load --base_model after client logs in, to be able to free GPU memory when model is swapped.

- **input_lines** (*int*): 
    How many input lines to show for chat box (>1 forces shift-enter for submit, else enter is submit).

- **large_file_count_mode** (*bool*): 
    Whether to force manual update to UI of drop-downs, good idea if millions of chunks or documents.

- **auth** (*list*): 
    Gradio auth for launcher in the form [(user1, pass1), (user2, pass2), ...]. Examples:
    - `--auth=[('jon','password')]` with no spaces
    - `--auth="[('jon', 'password)())(')]"` so any special characters can be used
    - `--auth=auth.json` to specify persisted state file with name auth.json (auth_filename then not required)
    - `--auth=''` will use default auth.json as file name for persisted state file (auth_filename good idea to control location)
    - `--auth=None` will use no auth, but still keep track of auth state, just not from logins

- **hyde_level** (*int*): 
    HYDE level for HYDE approach (https://arxiv.org/abs/2212.10496).
    - `0`: No HYDE.
    - `1`: Use non-document-based LLM response and original query for embedding query.
    - `2`: Use document-based LLM response and original query for embedding query.
    - `3+`: Continue iterations of embedding prior answer and getting new response.

- **hyde_show_only_final** (*bool*):  
    Whether to show only the last result of HYDE, not intermediate steps.

- **visible_models** (*list or None*): 
    Which models in model_lock list to show by default. Takes integers of position in model_lock (model_states) list or strings of base_model names. Ignored if model_lock not used. For nochat API, this is single item within a list for model by name or by index in model_lock. If None, then just use the first model in model_lock list. If model_lock not set, use the model selected by CLI --base_model etc. Note that unlike h2ogpt_key, this visible_models only applies to this running h2oGPT server, and the value is not used to access the inference server. If need a visible_models for an inference server, then use --model_lock and group together.

- **max_visible_models** (*int*): 
    Maximum visible models to allow to select in UI.

- **max_raw_chunks** (*int*): 
    Maximum number of chunks to show in UI when asking for raw DB text from documents/collection.

- **extra_model_options** (*str*): 
    Extra models to show in the list in Gradio.


- **langchain_mode** (*str*): 
    Data source to include. Choose "UserData" to only consume files from `make_db.py`. 
    If not passed, then chosen to be first `langchain_modes`, else `langchain_mode->Disabled` is set if no `langchain_modes` either. 
    WARNING: `wiki_full` requires extra data processing via `read_wiki_full.py` and requires really good workstation to generate db, unless already present.

- **user_path** (*str*): 
    User path to glob from to generate db for vector search, for 'UserData' langchain mode. 
    If already have db, any new/changed files are added automatically if path set, does not have to be the same path used for prior db sources.

- **langchain_modes** (*list*): 
    DBs to generate at launch to be ready for LLM. 
    Apart from additional user-defined collections, can include ['wiki', 'wiki_full', 'UserData', 'MyData', 'github h2oGPT', 'DriverlessAI docs']. 
    But `wiki_full` is expensive and requires preparation. 
    To allow personal space only live in session, add 'MyData' to list. 
    Default: If only want to consume local files, e.g. prepared by `make_db.py`, only include ['UserData']. 
    If have own user modes, need to add these here or add in UI.

- **langchain_mode_paths** (*dict*): 
    Dict of `langchain_mode` keys and disk path values to use for source of documents. 
    E.g. "{'UserData2': 'userpath2'}". 
    A disk path can be None, e.g. --langchain_mode_paths="{'UserData2': None}" even if existing DB, to avoid new documents being added from that path, source links that are on disk still work. 
    If `--user_path` was passed, that path is used for 'UserData' instead of the value in this dict.

- **langchain_mode_types** (*dict*): 
    Dict of `langchain_mode` keys and database types. 
    E.g. python generate.py --base_model=llama --langchain_modes=['TestData'] --langchain_mode_types="{'TestData':'shared'}". 
    The type is attempted to be inferred if the directory already exists, then don't have to pass this.

- **langchain_action** (*str*): 
    Mode langchain operations on documents.
    Options:
        - Query: Make query of document(s)
        - Summarize or Summarize_map_reduce: Summarize document(s) via map_reduce
        - Summarize_all: Summarize document(s) using entire document at once
        - Summarize_refine: Summarize document(s) using entire document, and try to refine before returning summary
        - Extract: Extract information from document(s) via map (no reduce)
    Currently enabled are Query, Summarize, and Extract.
    Summarize is a "map reduce" and extraction is "map". That is, map returns a text output (roughly) per input item, while reduce reduces all maps down to single text output.
    The "roughly" refers to the fact that if one has docs_token_handling='split_or_merge', then we split or merge chunks, so you will get a map for some optimal-sized chunks given the model size. If you choose docs_token_handling='chunk', then you get back a map for each chunk you give, but you should ensure the model token limit is not exceeded yourself.
    Summarize is useful when wanting to reduce down to single text, while Extract is useful when you want to operate the prompt on blocks of data and get back a result per block.

- **langchain_agents** (*list*): 
    Which agents to use.
    Options: 'search' (Use Web Search as context for LLM response, e.g. SERP if have `SERPAPI_API_KEY` in env)

- **visible_langchain_actions** (*bool*): 
    Which actions to allow.

- **visible_langchain_agents** (*bool*): 
    Which agents to allow.

- **document_subset** (*str*): 
    Default document choice when taking a subset of the collection.

- **document_choice** (*str*): 
    Chosen document(s) by internal name. 'All' means use all docs.

- **document_source_substrings** (*list*): 
    Substrings in the list to search in source names in metadata for chroma dbs.

- **document_source_substrings_op** (*str*): 
    'and' or 'or' for source search words.

- **document_content_substrings** (*list*): 
    Substrings in the list to search in content for chroma dbs.

- **document_content_substrings_op** (*str*): 
    'and' or 'or' for content search words.

- **use_llm_if_no_docs** (*bool*): 
    Whether to use LLM even if no documents, when `langchain_mode=UserData` or `MyData` or custom.

- **db_type** (*str*): 
    Type of database to use.
    Options:
        - 'faiss': in-memory database
        - 'chroma': for chroma >= 0.4
        - 'chroma_old': for chroma < 0.4 (recommended for large collections)
        - 'weaviate': for persisted on disk
        - 'qdrant': for a Qdrant server or an in-memory instance

- **hf_embedding_model** (*str*): 
    Which HF embedding model to use for the vector database.
    Default is instructor-large with 768 parameters per embedding if have GPUs, else all-MiniLM-L6-v2 if no GPUs.
    Can also choose simpler model with 384 parameters per embedding: "sentence-transformers/all-MiniLM-L6-v2".
    Can also choose even better embedding with 1024 parameters: 'hkunlp/instructor-xl'.
    We support automatically changing embeddings for chroma, with a backup of db made if this is done.

- **answer_with_sources** (*bool*): 
    Whether to determine (and return) sources.

- **append_sources_to_answer** (*bool*): 
    Whether to place source information in the chat response (ignored by LLM). Always disabled for API.

- **append_sources_to_chat** (*bool*): 
    Whether to place sources information in the chat response but in a separate chat turn (ignored by LLM). Always disabled for API.

- **pre_prompt_query** (*str*): 
    Prompt before documents to query. If `None`, then use internal defaults.

- **prompt_query** (*str*): 
    Prompt after documents to query. If `None`, then use internal defaults.

- **pre_prompt_summary** (*str*): 
    Prompt before documents to summarize/extract from. If `None`, then use internal defaults.

- **prompt_summary** (*str*): 
    Prompt after documents to summarize/extract from. If `None`, then use internal defaults.
    For summarize/extract, it is normal to have an empty query (nothing added in "ask anything" in the UI or empty string in the API).
    If a query is passed, the template is "Focusing on %s, %s" % (query, prompt_summary).
    If both query and input are passed, the template is "Focusing on %s, %s, %s" % (query, input, prompt_summary).

- **hyde_llm_prompt** (*str*): 
    Hyde prompt for the first step when using LLM.

- **allow_upload_to_my_data** (*bool*): 
    Whether to allow file uploads to update personal vector db.

- **enable_url_upload** (*bool*): 
    Whether to allow upload from URL.

- **enable_text_upload** (*bool*): 
    Whether to allow upload of text.

- **enable_sources_list** (*bool*): 
    Whether to allow list (or download for non-shared db) of list of sources for chosen db.

- **chunk** (*bool*): 
    Whether to chunk data (True unless know data is already optimally chunked).

- **chunk_size** (*int*): 
    Size of chunks, with typically top-4 passed to LLM, so needs to be in context length.

- **top_k_docs** (*int*): 
    For `langchain_action` query: number of chunks to give LLM.
    -1 : auto-fills context up to max_seq_len
    For `langchain_action` summarize/extract: number of document parts, like pages for PDF.
    There's no such thing as chunks for summarization.
    -1 : auto-fills context up to max_seq_len

- **docs_ordering_type** (*str*):
    Type of ordering of docs.
    - 'best_first': Order by score so score is worst match near prompt.
    - 'best_near_prompt' or 'reverse_sort': Reverse docs order so most relevant is closest to question.
      Best choice for sufficiently smart model, and truncation occurs for oldest context, so best then too.
      But smaller 6_9 models fail to use newest context and can get stuck on old information.
    - '' or None (i.e. default) or 'reverse_ucurve_sort': Sort so most relevant is either near start or near end.
      Best to avoid "lost in middle" as well as avoid hallucinating off starting content that LLM focuses on a lot.

- **auto_reduce_chunks** (*bool*):
    Whether to automatically reduce `top_k_docs` to fit context given prompt.

- **max_chunks** (*int*):
    If `top_k_docs=-1`, maximum number of chunks to allow.

- **use_pymupdf** (*str*):
    Enable PyMUPDF loader. 'auto' means use first, use others if they are 'auto' if no result.

- **use_unstructured_pdf** (*str*):
    Enable Unstructured PDF loader. 'auto' means use if pymupdf fails to get doc result.

- **use_pypdf** (*str*):
    Enable PyPDF loader. 'auto' means use if unstructured fails to get doc result.

- **enable_pdf_ocr** (*str*):
    Control OCR for PDF files.
    - 'auto': Only use OCR if normal text extraction fails. Useful for pure image-based PDFs with text.
    - 'on': Always perform OCR as additional parsing of same documents.
    - 'off': Don't perform OCR (e.g., because it's slow even if 'auto' only would trigger if nothing else worked).

- **enable_pdf_doctr** (*str*):
    Whether to support doctr on PDFs.
    - 'auto': Use doctr if failed to get doc result so far.

- **try_pdf_as_html** (*bool*):
    Try "PDF" as if HTML file, in case web link has .pdf extension but really is just HTML.

- **enable_ocr** (*bool*):
    Whether to support OCR on images.

- **enable_doctr** (*bool*):
    Whether to support doctr on images (using OCR better than enable_ocr=True).

- **enable_pix2struct** (*bool*):
    Whether to support pix2struct on images for captions.

