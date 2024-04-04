Fonction Main (gen.py)
=============

Documentation générale de la fonction Main de gen.py contenant tous les paramètres modifiables de l'application

Paramètres:

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
    
- **use_flash_attention_2** (*bool*): 
    Whether to try to use flash attention 2 if available when loading HF models.
    Warning: We have seen nans and type mismatches with flash-attn==2.3.4 installed and this enabled, even for other models like embedding model that is unrelated to primary models.
    
- **load_gptq** (*str*): 
    To load model with GPTQ, put model_basename here, e.g. 'model' for TheBloke models.
    
- **use_autogptq** (*bool*): 
    Whether to use AutoGPTQ (True) or HF Transformers (False). Some models are only supported by one or the other.
    
- **load_awq** (*str*): 
    Load model with AWQ, e.g. 'model' for TheBloke models.
    
- **load_exllama** (*bool*): 
    Whether to use exllama (only applicable to LLaMa1/2 models with 16-bit or GPTQ).
    
- **use_safetensors** (*bool*): 
    To use safetensors version (assumes file/HF points to safe tensors version).
    
- **revision** (*str*): 
    Which HF revision to use.
    
- **use_gpu_id** (*bool*): 
    Whether to control devices with gpu_id. If False, then spread across GPUs.
    
- **base_model** (*str*): 
    Model HF-type name. If use --base_model to preload model, cannot unload in gradio in models tab.
    
- **tokenizer_base_model** (*str*): 
    Tokenizer HF-type name. Usually not required, inferred from base_model. If model is private or doesn't exist as HF model, can use "tiktoken" and pass max_seq_len and (if different) max_output_seq_len. For inference servers like OpenAI etc. if have model name, we use tiktoken with known input/output sequence lengths.
    
- **lora_weights** (*str*): 
    LORA weights path/HF link.
    
- **gpu_id** (*int*): 
    If use_gpu_id, then use gpu_id for cuda device ID, or auto mode if gpu_id != -1.
    
- **compile_model** (*bool*): 
    Whether to compile the model.
    
- **use_cache** (*bool*): 
    Whether to use caching in model (some models fail when multiple threads use).
    
- **inference_server** (*str*): 
    Consume base_model as type of model at this address.

    Address can be text-generation-server hosting that base_model.
    For example:
    
    - ``python generate.py --inference_server="http://192.168.1.46:6112" --base_model=HuggingFaceH4/zephyr-7b-beta``
    
    For a gradio server, use the same address as the TGI server. We infer if it's TGI or Gradio.
    For example:
    
    - ``python generate.py --inference_server="http://192.168.1.46:7860" --base_model=HuggingFaceH4/zephyr-7b-beta``
    
    For auth protected gradio, do:
    For example:
    
    - ``python generate.py --inference_server="http://192.168.1.46:7860:user:password" --base_model=HuggingFaceH4/zephyr-7b-beta``
    
    If you don't want to specify the port, do:
    For example:
    
    - ``python generate.py --inference_server="https://gpt.h2o.ai:None:user:password" --base_model=HuggingFaceH4/zephyr-7b-beta``

    Address can also be "openai_chat" or "openai" for OpenAI API.

    Address can also be "openai_azure_chat" or "openai_azure" for Azure OpenAI API.
    For example:
    
    - ``python generate.py --inference_server="openai_chat" --base_model=gpt-3.5-turbo``
    - ``python generate.py --inference_server="openai" --base_model=text-davinci-003``
    - ``python generate.py --inference_server="openai_azure_chat:<deployment_name>:<baseurl>:<api_version>:<access key>" --base_model=gpt-3.5-turbo``
    - ``python generate.py --inference_server="openai_azure:<deployment_name>:<baseurl>:<api_version>:<access key>" --base_model=text-davinci-003``

    Optionals (Replace with None or just leave empty but keep :):

    - `<deployment_name>`: Some deployment name
    - `<baseurl>`: For example, "<endpoint>.openai.azure.com" for some <endpoint> without https://
    - `<api_version>`: Some API version, e.g., 2023-05-15

    Address can also be for vLLM:

    - Use: "vllm:IP:port" for OpenAI-compliant vLLM endpoint
    - Use: "vllm_chat:IP:port" for OpenAI-Chat-compliant vLLM endpoint
    - Use: "vllm:http://IP:port/v1" for OpenAI-compliant vLLM endpoint
    - Use: "vllm_chat:http://IP:port/v1" for OpenAI-Chat-compliant vLLM endpoint
    - Use: "vllm:https://IP/v1" for OpenAI-compliant vLLM endpoint
    - Use: "vllm_chat:https://IP/v1" for OpenAI-Chat-compliant vLLM endpoint

    For example, for non-standard URL and API key for vllm, one would do:

    - ``vllm_chat:https://vllm.h2o.ai:None:/1b1219f7-4bb4-43e9-881f-fa8fa9fe6e04/v1:1234ABCD``
    where vllm.h2o.ai is the DNS name of the IP, None means no extra port, so will be dropped from base_url when using API, /1b1219f7-4bb4-43e9-881f-fa8fa9fe6e04/v1 is the url of the "page" to access, and 1234ABCD is the API key

    - ``vllm_chat:https://vllm.h2o.ai:5001:/1b1219f7-4bb4-43e9-881f-fa8fa9fe6e04/v1:1234ABCD``
    where vllm.h2o.ai is the DNS name of the IP, 5001 is the port, /1b1219f7-4bb4-43e9-881f-fa8fa9fe6e04/v1 is the url of the "page" to access, and 1234ABCD is the API key

    Or for groq, can use OpenAI API like:

    - GROQ IS BROKEN FOR OPENAI API: ``vllm:https://api.groq.com/openai:None:/v1:<api key>'``
    with: other model_lock or CLI options: {'inference_server': 'vllm:https://api.groq.com/openai:None:/v1:<api key>', 'base_model':'mixtral-8x7b-32768', 'visible_models':'mixtral-8x7b-32768', 'max_seq_len': 31744, 'prompt_type':'plain'}
    i.e.ensure to use 'plain' prompt, not mixtral.

    For groq:

    - groq and ensures set env GROQ_API_KEY or ``groq:<api key>``
    with: other model_lock or CLI options: {'inference_server': 'groq:<api key>', 'base_model':'mixtral-8x7b-32768', 'visible_models':'mixtral-8x7b-32768', 'max_seq_len': 31744, 'prompt_type':'plain'}

    Or Address can be replicate:

    - Use: ``--inference_server=replicate:<model name string>`` will use a Replicate server, requiring a Replicate key.
    e.g. <model name string> looks like "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

    Or Address can be for AWS SageMaker:

    - Use: "sagemaker_chat:<endpoint name>" for chat models that AWS sets up as dialog
    - Use: "sagemaker:<endpoint name>" for foundation models that AWS only text as inputs

    Or Address can be for Anthropic Claude.  Ensure key is set in env ANTHROPIC_API_KEY

    - Use: "anthropic"
    E.g. ``--base_model=claude-2.1 --inference_server=anthropic``

    Or Address can be for Google Gemini.  Ensure key is set in env GOOGLE_API_KEY

    - Use: "google"
    E.g. ``--base_model=gemini-pro --inference_server=google``

    Or Address can be for MistralAI.  Ensure key is set in env MISTRAL_API_KEY

    - Use: "mistralai"
    E.g. ``--base_model=mistral-medium --inference_server=mistralai``

- **regenerate_clients** (*bool*): 
    Whether to regenerate client every LLM call or use start-up version.

    Benefit of doing each LLM call is timeout can be controlled to max_time in expert settings, else we use default of 600s.

    Maybe risky, some lack of thread safety: https://github.com/encode/httpx/discussions/3043, so disabled
    Because gradio clients take a long time to start-up, we don't ever regenerate them each time (including llava models).

- **regenerate_gradio_clients** (*bool*): 
    Whether to also regenerate gradio clients (slow).

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

- **llamacpp_path** (*str*): 
    Location to store downloaded gguf or load list of models from. Note HF models go into hf cache folder, and gpt4all models go into their own cache folder. Can override with ENV LLAMACPP_PATH.

- **llamacpp_dict** (*dict*): 
    - n_gpu_layers: for llama.cpp based models, number of GPU layers to offload (default is all by using large value).
    - use_mlock: when using `llama.cpp` based CPU models, for computers with low system RAM or slow CPUs, recommended False.
    - n_batch: Can make smaller to 128 for slower low-memory CPU systems.
    - n_gqa: Required to be 8 for LLaMa 70B.
    - ... etc. anything that could be passed to llama.cpp or GPT4All models. e.g. python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path --llamacpp_dict="{'n_gpu_layers':25,'n_batch':128}"

- **model_path_llama** (*str*): 
    Model path or URL (for auto-download).

- **model_name_gptj** (*str*): 
    Model path or URL (for auto-download).

- **model_name_gpt4all_llama** (*str*): 
    Model path or URL (for auto-download).

- **model_name_exllama_if_no_config** (*str*): 
    exllama model's full path for model, tokenizer, generator for use when no HuggingFace config.

- **exllama_dict** (*dict*): 
    for setting various things for Exllama class: 
    - compress_pos_emb
    - set_auto_map
    - gpu_peer_fix
    - alpha_value
    - matmul_recons_thd
    - fused_mlp_thd
    - sdp_thd
    - fused_attn
    - matmul_fused_remap
    - rmsnorm_no_half2
    - rope_no_half2
    - matmul_no_half2
    - silu_no_half2
    - concurrent_streams

    E.g. to set memory to be split across 2 GPUs, use --exllama_dict="{'set_auto_map':20,20}"

- **gptq_dict** (*dict*): 
    Choices for AutoGPTQ.

    - **inject_fused_attention** (*bool*): Whether to inject fused attention.
    - **disable_exllama** (*bool*): Whether to disable ExLLAMA.
    - **use_triton** (*bool*): Whether to use Triton.

- **attention_sinks** (*bool*): 
    Whether to enable attention sinks.

- **sink_dict** (*dict*): 
    Dict of options for attention sinks.

    - **window_length** (*int*): Length of the window.
    - **num_sink_tokens** (*int*): Number of sink tokens.

- **hf_model_dict** (*dict*): 
    Dict of options for HF models using transformers.

- **truncation_generation** (*bool*): 
    Whether (for torch) to terminate generation once reach context length of model. For some models, perplexity becomes critically large beyond context.

- **model_lock** (*list of dict*): 
    Lock models to specific combinations, for ease of use and extending to many models. Only used if gradio = True. List of dicts, each dict has base_model, tokenizer_base_model, lora_weights, inference_server, prompt_type, and prompt_dict. If all models have the same prompt_type and prompt_dict, you can still specify that once in the CLI outside model_lock as the default for the dict. You can specify model_lock instead of those items on the CLI. As with the CLI itself, base_model can infer prompt_type and prompt_dict if in prompter.py. Also, tokenizer_base_model and lora_weights are optional. Also, inference_server is optional if loading the model from the local system. All models provided will automatically appear in compare model mode. Model loading-unloading and related choices will be disabled. Model/lora/server adding will be disabled.

- **model_lock_columns** (*int*): 
    How many columns to show if locking models (and so showing all at once). If None, then defaults to up to 3. if -1, then all goes into 1 row. Maximum value is 4 due to non-dynamic gradio rendering elements.

- **model_lock_layout_based_upon_initial_visible** (*bool*): 
    Whether to base any layout upon visible models (True) or upon all possible models. Gradio does not allow dynamic objects, so all layouts are preset, and these are two reasonable options. False is best when there are many models and user excludes middle ones as being visible.

- **fail_if_cannot_connect** (*bool*): 
    If doing model locking (e.g. with many models), fail if True. Otherwise, ignore. Useful when many endpoints and want to just see what works, but still have to wait for timeout.

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

- **local_files_only** (*bool*): 
    Whether to only use local files instead of doing to HF for models.

- **resume_download** (*bool*): 
    Whether to resume downloads from HF for models.

- **use_auth_token** (*bool*): 
    Whether to use HF auth token (requires CLI did huggingface-cli login before).

- **trust_remote_code** (*bool*): 
    Whether to trust any code needed for HF model.

- **rope_scaling** (*str*): 
    For HF transformers model: scaling for rope-based models. For long context models that have been tuned for a specific size, you have to only use that specific size by setting the `--rope_scaling` exactly correctly.
    
    Example usage:
        - `--rope_scaling="{'type':'dynamic', 'factor':4}"`
        - `--rope_scaling="{'type':'linear', 'factor':4}"`
        - `python generate.py --rope_scaling="{'type':'linear','factor':4}" --base_model=lmsys/vicuna-13b-v1.5-16k --hf_embedding_model=sentence-transformers/all-MiniLM-L6-v2 --load_8bit=True --langchain_mode=UserData --user_path=user_path --prompt_type=vicuna11 --h2ocolors=False`

    For exllama model: `--rope_scaling="{'alpha_value':4}"`. This automatically scales max_seq_len for exllama.

- **max_seq_len** (*int*): 
    Manually set maximum sequence length for the LLM.

- **max_output_seq_len** (*int*): 
    Manually set maximum output length for the LLM.

- **offload_folder** (*str*): 
    Path for spilling model onto disk.

- **src_lang** (*str or None*): 
    Source languages to include if doing translation (None = all).

- **tgt_lang** (*str or None*): 
    Target languages to include if doing translation (None = all).

- **prepare_offline_level** (*int*): 
    Whether to just prepare for offline use, do not go into CLI, eval, or Gradio run modes.

    - ``0``: No preparation.
    - ``1``: Prepare just h2oGPT with the exact same setup as passed to CLI and ensure all artifacts for h2oGPT alone added to ~/.cache/.
    - ``2``: Prepare h2oGPT + all inference servers so h2oGPT + inference servers can use the ~/.cache/.

- **cli** (*bool*): 
    Whether to use CLI (non-Gradio) interface.

- **cli_loop** (*bool*): 
    Whether to loop for CLI (False usually only for testing).

- **gradio** (*bool*): 
    Whether to enable Gradio, or to enable benchmark mode.

- **openai_server** (*bool*): 
    Whether to launch OpenAI proxy server for local Gradio server. Disabled if API is disabled or --auth=closed.

- **openai_port** (*int*): 
    Port for OpenAI proxy server.

- **gradio_offline_level** (*int*): 
    If greater than 0, then change fonts so full offline.

    - ``== 1``: Means backend won't need internet for fonts, but front-end UI might if font not cached.
    - ``== 2``: Means backend and frontend don't need internet to download any fonts. Note: Some things always disabled include HF telemetry, Gradio telemetry, ChromaDB posthog that involve uploading. This option further disables Google Fonts for downloading, which is less intrusive than uploading, but still required in air-gapped case. The fonts don't look as nice as Google Fonts, but ensure full offline behavior. Also set ``--share=False`` to avoid sharing a Gradio live link.

- **server_name** (*str*): 
    IP to use. In Linux, 0.0.0.0 is a good choice so exposed to outside host, else for only local use 127.0.0.1. For Windows/MAC, 0.0.0.0 or 127.0.0.1 will work, but may need to specify the actual LAN IP address for other LAN clients to see.

- **share** (*bool*): 
    Whether to share the Gradio app with a sharable URL.

- **open_browser** (*bool*): 
    Whether to automatically open a browser tab with Gradio UI.

- **close_button** (*bool*): 
    Whether to show close button in system tab (if not public).

- **shutdown_via_api** (*bool*): 
    Whether to allow shutdown via API.

- **root_path** (*str*): 
    The root path (or "mount point") of the application, if it's not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application. For example, if the application is served at "https://example.com/myapp", the `root_path` should be set to "/myapp".

- **ssl_verify** (*str*): 
    Passed to Gradio launch.

- **ssl_keyfile** (*str*): 
    Passed to Gradio launch.

- **ssl_certfile** (*str*): 
    Passed to Gradio launch.

- **ssl_keyfile_password** (*str*): 
    Passed to Gradio launch.

- **chat** (*bool*): 
    Whether to enable chat mode with chat history.

- **chat_conversation** (*list of tuples*): 
    List of tuples of (human, bot) conversation pre-appended to existing chat when using instruct/chat models. Requires also `add_chat_history_to_context = True`. It does *not* require `chat=True`, so works with nochat_api etc.

- **text_context_list** (*list of str*): 
    List of strings to add to context for non-database version of document Q/A for faster handling via API etc. Forces LangChain code path and uses as many entries in list as possible given `max_seq_len`, with first assumed to be most relevant and to go near prompt.

- **stream_output** (*bool*): 
    Whether to stream output.

- **async_output** (*bool*): 
    Whether to do asyncio handling.

- **num_async** (*int*): 
    Number of simultaneously allowed asyncio calls to make for async_output. Too many will overload the inference server, too few will be too slow.

- **show_examples** (*bool*): 
    Whether to show clickable examples in Gradio.

- **verbose** (*bool*): 
    Whether to show verbose prints.

- **h2ocolors** (*bool*): 
    Whether to use H2O.ai theme.

- **dark** (*bool*): 
    Whether to use dark mode for UI by default (still controlled in UI).

- **height** (*int*): 
    Height of chat window.

- **render_markdown** (*bool*): 
    Whether to render markdown in chatbot UI. In some cases this distorts the rendering. [More Info](https://github.com/gradio-app/gradio/issues/4344#issuecomment-1771963021)

- **show_lora** (*bool*): 
    Whether to show LORA options in UI (expert so can be hard to understand).

- **show_llama** (*bool*): 
    Whether to show LLaMa.cpp/GPT4All options in UI (only likely useful if have weak GPUs).

- **show_gpt4all** (*bool*): 
    Whether to show GPT4All models in UI (not often useful, llama.cpp models best).

- **login_mode_if_model0** (*bool*): 
    Set to True to load --base_model after client logs in, to be able to free GPU memory when model is swapped.

- **block_gradio_exit** (*bool*): 
    Whether to block Gradio exit (used for testing).

- **concurrency_count** (*int*): 
    Gradio concurrency count (1 is optimal for local LLMs to avoid sharing cache that messes up models, else 64 is used if hosting remote inference servers only).

- **api_open** (*bool*): 
    If False, don't let API calls skip Gradio queue.

- **allow_api** (*bool*): 
    Whether to allow API calls at all to Gradio server.

- **input_lines** (*int*): 
    How many input lines to show for chat box (>1 forces shift-enter for submit, else enter is submit).

- **gradio_size** (*str*): 
    Overall size of text and spaces: "xsmall", "small", "medium", "large". Small useful for many chatbots in model_lock mode.

- **show_copy_button** (*bool*): 
    Whether to show copy button for chatbots.

- **large_file_count_mode** (*bool*): 
    Whether to force manual update to UI of drop-downs, good idea if millions of chunks or documents.

- **gradio_ui_stream_chunk_size** (*int or None*): 
    Number of characters to wait before pushing text to UI. None is default, which is 0 when not doing model lock. Else 20 by default. 20 is a reasonable value for fast models and fast systems when handling several models at once. Choose 0 to disable (this disables use of `gradio_ui_stream_chunk_min_seconds` and `gradio_ui_stream_chunk_seconds` too). Workaround for these bugs that lead to UI being overwhelmed under various cases: [Issue 5914](https://github.com/gradio-app/gradio/issues/5914) and [Issue 6609](https://github.com/gradio-app/gradio/issues/6609).

- **gradio_ui_stream_chunk_min_seconds** (*float*): 
    Number of seconds before allowing yield to avoid spamming yields at a rate the user would not care about, regardless of chunk_size.

- **gradio_ui_stream_chunk_seconds** (*float*): 
    Number of seconds to yield regardless of reaching `gradio_ui_stream_chunk_size` as long as something to yield. Helps case when streaming is slow and want to see progress at least every couple seconds.

- **gradio_api_use_same_stream_limits** (*bool*): 
    Whether to use the same streaming limits as UI for API.

- **gradio_upload_to_chatbot** (*bool*): 
    Whether to show upload in chatbots.

- **gradio_upload_to_chatbot_num_max** (*int*): 
    Max number of things to add to chatbot.

- **gradio_errors_to_chatbot** (*bool*): 
    Whether to show errors in Accordion in chatbot or just in exceptions in each tab.

- **pre_load_embedding_model** (*bool*): 
    Whether to preload embedding model for shared use across DBs and users (multi-thread safe only).

- **embedding_gpu_id** (*str*): 
    Which GPU to place embedding model on. Only used if preloading embedding model. If 'auto', then use first device as is default. If 'cpu' or some other string like 'mps', then use that as device name.

- **auth** (*list*): 
    Gradio auth for launcher in the form [(user1, pass1), (user2, pass2), ...]. Examples:
    - `--auth=[('jon','password')]` with no spaces
    - `--auth="[('jon', 'password)())(')]"` so any special characters can be used
    - `--auth=auth.json` to specify persisted state file with name auth.json (auth_filename then not required)
    - `--auth=''` will use default auth.json as file name for persisted state file (auth_filename good idea to control location)
    - `--auth=None` will use no auth, but still keep track of auth state, just not from logins

- **auth_filename** (*str*): 
    Set auth filename, used only if --auth= was passed list of user/passwords.

- **auth_access** (*str*): 
    'open': Allow new users to be added. 'closed': Stick to existing users.

- **auth_freeze** (*bool*): 
    Whether to freeze authentication based upon the current file, no longer update file.

- **auth_message** (*str*): 
    Message to show if having users login, fixed if passed, else dynamic internally.

- **google_auth** (*bool*): 
    Whether to use Google auth.

- **guest_name** (*str*): 
    Guest name if using auth and have open access. If '', then no guest allowed even if open access, then all databases for each user always persisted.

- **enforce_h2ogpt_api_key** (*bool*): 
    Whether to enforce h2oGPT token usage for API.

- **enforce_h2ogpt_ui_key** (*bool*): 
    Whether to enforce h2oGPT token usage for UI (same keys as API assumed).

- **h2ogpt_api_keys** (*list or str*): 
    List of tokens allowed for API access or file accessed on demand for JSON of list of keys.

- **h2ogpt_key** (*str*): 
    E.g. can be set when accessing Gradio h2oGPT server from local Gradio h2oGPT server that acts as a client to that inference server. Only applied for API at runtime when API accesses using Gradio inference_server are made.

- **extra_allowed_paths** (*list*): 
    List of strings for extra allowed paths users could access for file viewing/downloading. '.' can be used but be careful what that exposes. Note by default all paths in `langchain_mode_paths` given at startup are allowed.

- **blocked_paths** (*list*): 
    Any blocked paths to add for Gradio access for file viewing/downloading.

- **max_max_time** (*float*): 
    Maximum max_time for Gradio slider.

- **max_max_new_tokens** (*int*): 
    Maximum max_new_tokens for Gradio slider.

- **min_max_new_tokens** (*int*): 
    Minimum of max_new_tokens, when auto-scaling down to handle more docs/prompt, but still let generation have some tokens.

- **max_input_tokens** (*int*): 
    Max input tokens to place into model context for each LLM call. -1 means auto, fully fill context for query, and fill by original document chunk for summarization. >=0 means use that to limit context filling to that many tokens.

- **max_total_input_tokens** (*int*): 
    Like max_input_tokens but instead of per LLM call, applies across all LLM calls for single summarization/extraction action.

- **docs_token_handling** (*str*): 
    - `'chunk'` means fill context with top_k_docs (limited by max_input_tokens or model_max_len) chunks for query or top_k_docs original document chunks summarization.
    - `None` or `'split_or_merge'` means same as 'chunk' for query, while for summarization merges documents to fill up to max_input_tokens or model_max_len tokens.

- **docs_joiner** (*str or None*): 
    String to join lists of text when doing split_or_merge. `None` means '\n\n'.

- **hyde_level** (*int*): 
    HYDE level for HYDE approach (https://arxiv.org/abs/2212.10496).
    - `0`: No HYDE.
    - `1`: Use non-document-based LLM response and original query for embedding query.
    - `2`: Use document-based LLM response and original query for embedding query.
    - `3+`: Continue iterations of embedding prior answer and getting new response.

- **hyde_template** (*str or None*): 
    - `None`, `'None'`, `'auto'` uses internal value and enable.
    - `' {query} '` is minimal template one can pass.

- **hyde_show_only_final** (*bool*):  
    Whether to show only the last result of HYDE, not intermediate steps.

- **hyde_show_intermediate_in_accordion** (*bool*): 
    Whether to show intermediate HYDE, but inside HTML accordion.

- **visible_models** (*list or None*): 
    Which models in model_lock list to show by default. Takes integers of position in model_lock (model_states) list or strings of base_model names. Ignored if model_lock not used. For nochat API, this is single item within a list for model by name or by index in model_lock. If None, then just use the first model in model_lock list. If model_lock not set, use the model selected by CLI --base_model etc. Note that unlike h2ogpt_key, this visible_models only applies to this running h2oGPT server, and the value is not used to access the inference server. If need a visible_models for an inference server, then use --model_lock and group together.

- **max_visible_models** (*int*): 
    Maximum visible models to allow to select in UI.

- **visible_ask_anything_high** (*bool*): 
    Whether the ask anything block goes near the top or near the bottom of the UI Chat.

- **visible_visible_models** (*bool*): 
    Whether the visible models drop-down is visible in the UI.

- **visible_submit_buttons** (*bool*): 
    Whether submit buttons are visible when the UI first comes up.

- **visible_side_bar** (*bool*): 
    Whether the left sidebar is visible when the UI first comes up.

- **visible_doc_track** (*bool*): 
    Whether the left sidebar's document tracking is visible when the UI first comes up.

- **visible_chat_tab** (*bool*): 
    Whether the chat tab is visible.

- **visible_doc_selection_tab** (*bool*): 
    Whether the document selection tab is visible.

- **visible_doc_view_tab** (*bool*): 
    Whether the document view tab is visible.

- **visible_chat_history_tab** (*bool*): 
    Whether the chat history tab is visible.

- **visible_expert_tab** (*bool*): 
    Whether the expert tab is visible.

- **visible_models_tab** (*bool*): 
    Whether the models tab is visible.

- **visible_system_tab** (*bool*): 
    Whether the system tab is visible.

- **visible_tos_tab** (*bool*): 
    Whether the ToS tab is visible.

- **visible_login_tab** (*bool*): 
    Whether the Login tab is visible (needed for persistence or to enter key for UI access to models and ingestion).

- **visible_hosts_tab** (*bool*): 
    Whether the hosts tab is visible.

- **chat_tables** (*bool*): 
    Just show Chat as a block without a tab (useful if you want only chat view).

- **visible_h2ogpt_links** (*bool*): 
    Whether GitHub stars, URL are visible.

- **visible_h2ogpt_qrcode** (*bool*): 
    Whether QR code is visible.

- **visible_h2ogpt_logo** (*bool*): 
    Whether the central logo is visible.

- **visible_chatbot_label** (*bool*): 
    Whether to show label in chatbot (e.g., if only one model for own purpose, then can set to False).

- **visible_all_prompter_models** (*bool*): 
    Whether to show all prompt_type_to_model_name items or just curated ones.

- **visible_curated_models** (*bool*): 
    Whether to show curated models (useful to see few good options).

- **actions_in_sidebar** (*bool*): 
    Whether to show sidebar with actions in old style.

- **document_choice_in_sidebar** (*bool*): 
    Whether to show document choices in the sidebar. Useful if often changing picking specific document(s).

- **enable_add_models_to_list_ui** (*bool*): 
    Whether to show add model, lora, server to dropdown list. Disabled by default since it clutters Models tab in UI, and can just add a custom item directly in the dropdown.

- **max_raw_chunks** (*int*): 
    Maximum number of chunks to show in UI when asking for raw DB text from documents/collection.

- **pdf_height** (*str*): 
    Height of PDF viewer in UI.

- **avatars** (*bool*): 
    Whether to show avatars in chatbot.

- **add_disk_models_to_ui** (*bool*): 
    Whether to add HF cache models and llama.cpp models to UI.

- **page_title** (*str*): 
    Title of the web page. Default is "h2oGPT".

- **favicon_path** (*str*): 
    Path to favicon. Default is "h2oGPT" favicon.

- **visible_ratings** (*bool*): 
    Whether full review is visible, else just likable chatbots.

- **reviews_file** (*str*): 
    File to store reviews. Set to `reviews.csv` if `visible_ratings=True` if this isn't set.

- **sanitize_user_prompt** (*bool*): 
    Whether to remove profanity from user input (slows down input processing). Requires optional packages: `alt-profanity-check==1.2.2` and `better-profanity==0.7.0`.

- **sanitize_bot_response** (*bool*): 
    Whether to remove profanity and repeat lines from bot output (about 2x slower generation for long streaming cases due to `better_profanity` being slow).

- **extra_model_options** (*str*): 
    Extra models to show in the list in Gradio.

- **extra_lora_options** (*str*): 
    Extra LORA to show in the list in Gradio.

- **extra_server_options** (*str*): 
    Extra servers to show in the list in Gradio.

- **score_model** (*str*): 
    Which model to score responses. Options are: 
    - `None`: No response scoring.
    - `'auto'`: Auto mode. Use '' (no model) for CPU or 1 GPU, 'OpenAssistant/reward-model-deberta-v3-large-v2' for >=2 GPUs, because on CPU takes too much compute just for scoring response.

- **verifier_model** (*str*): 
    Model for verifier.

- **verifier_tokenizer_base_model** (*str*): 
    Tokenizer server for verifier. If empty/None, infer from the model.

- **verifier_inference_server** (*str*): 
    Inference server for verifier.

- **eval_filename** (*str*): 
    JSON file to use for evaluation. If `None`, it is `sharegpt`.

- **eval_prompts_only_num** (*int*): 
    For no Gradio benchmark, if using `eval_filename` prompts for eval instead of examples.

- **eval_prompts_only_seed** (*int*): 
    For no Gradio benchmark, seed for `eval_filename` sampling.

- **eval_as_output** (*bool*): 
    For no Gradio benchmark, whether to test `eval_filename` output itself.

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

- **detect_user_path_changes_every_query** (*bool*): 
    Whether to detect if any files changed or added every similarity search (by file hashes). 
    Expensive for a large number of files, so not done by default. By default, only detect changes during db loading.

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

- **force_langchain_evaluate** (*bool*): 
    Whether to force langchain LLM use even if not doing langchain, mostly for testing.

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

- **load_db_if_exists** (*bool*): 
    Whether to load chroma db if exists or re-generate db.

- **keep_sources_in_context** (*bool*): 
    Whether to keep URL sources in context, not helpful usually.

- **db_type** (*str*): 
    Type of database to use.
    Options:
        - 'faiss': in-memory database
        - 'chroma': for chroma >= 0.4
        - 'chroma_old': for chroma < 0.4 (recommended for large collections)
        - 'weaviate': for persisted on disk
        - 'qdrant': for a Qdrant server or an in-memory instance

- **use_openai_embedding** (*bool*): 
    Whether to use OpenAI embeddings for the vector database.

- **use_openai_model** (*bool*): 
    Whether to use OpenAI model for use with the vector database.

- **hf_embedding_model** (*str*): 
    Which HF embedding model to use for the vector database.
    Default is instructor-large with 768 parameters per embedding if have GPUs, else all-MiniLM-L6-v2 if no GPUs.
    Can also choose simpler model with 384 parameters per embedding: "sentence-transformers/all-MiniLM-L6-v2".
    Can also choose even better embedding with 1024 parameters: 'hkunlp/instructor-xl'.
    We support automatically changing embeddings for chroma, with a backup of db made if this is done.

- **migrate_embedding_model** (*bool*): 
    Whether to use `hf_embedding_model` embedding even if the database already had an embedding set.
    Used to migrate all embeddings to a new one, but will take time to re-embed.
    Default (`False`) is to use the prior embedding for existing databases, and only use `hf_embedding_model` for new databases.
    If had an old database without an embedding saved, then `hf_embedding_model` is also used.

- **auto_migrate_db** (*bool*): 
    Whether to automatically migrate any chroma<0.4 database from duckdb -> sqlite version.

- **cut_distance** (*float*): 
    Distance to cut off references with larger distances when showing references.
    `1.64` is good to avoid dropping references for all-MiniLM-L6-v2, but `instructor-large` will always show excessive references.
    For `all-MiniLM-L6-v2`, a value of `1.5` can push out even more references, or a large value of `100` can avoid any loss of references.

- **answer_with_sources** (*bool*): 
    Whether to determine (and return) sources.

- **append_sources_to_answer** (*bool*): 
    Whether to place source information in the chat response (ignored by LLM). Always disabled for API.

- **append_sources_to_chat** (*bool*): 
    Whether to place sources information in the chat response but in a separate chat turn (ignored by LLM). Always disabled for API.

- **show_accordions** (*bool*): 
    Whether to show accordion for document references in the chatbot UI.

- **top_k_docs_max_show** (*int*): 
    Maximum number of documents to show in the UI for sources.
    If web search is enabled, then this is modified to be max(top_k_docs_max_show, number of links used in search).

- **show_link_in_sources** (*bool*): 
    Whether to show URL link to the source document in references.

- **langchain_instruct_mode** (*bool*): 
    Whether to have langchain operate in instruct mode (`True`) or few-shot mode (`False`).
    Normally this might be decidable from `--prompt_type=plain`, but in some cases (like vllm_chat) we want the inference server to handle all prompting, so need to tell h2oGPT to use plain prompting, but don't want to change langchain behavior.

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

- **doc_json_mode** (*bool*): 
    Use system prompting approach with JSON input and output, e.g., for codellama or GPT-4.

- **metadata_in_context** (*str*): 
    Keys of metadata to include in LLM context for Query. 
    Options:
        - `'all'`: Include all metadata.
        - `'auto'`: Includes these keys: ['date', 'file_path', 'input_type', 'keywords', 'chunk_id', 'page', 'source', 'title', 'total_pages'].
        - `['key1', 'key2', ...]`: Include only these keys.
        NOTE: Not all parsers have all keys, only keys that exist are added to each document chunk.
    Example key-values that some PDF parsers make:
        - author = Zane Durante, Bidipta Sarkar, Ran Gong, Rohan Taori, Yusuke Noda, Paul Tang, Ehsan Adeli, Shrinidhi Kowshika Lakshmikanth, Kevin Schulman, Arnold Milstein, Demetri Terzopoulos, Ade Famoti, Noboru Kuno, Ashley Llorens, Hoi Vo, Katsu Ikeuchi, Li Fei-Fei, Jianfeng Gao, Naoki Wake, Qiuyuan Huang
        - chunk_id = 21
        - creationDate = D:20240209020045Z
        - creator = LaTeX with hyperref
        - date = 2024-02-11 23:58:11.929155
        - doc_hash = 5db1d548-7
        - file_path = /tmp/gradio/15ac25af8610f21b9ab55252f1944841727ba157/2402.05929.pdf
        - format = PDF 1.5
        - hashid = 3cfb31cea127c745c72554f4714105dd
        - head = An Interactive Agent Foundation Model
        - Figure 2. We
        - input_type = .pdf
        - keywords = Machine Learning, ICML
        - modDate = D:20240209020045Z
        - order_id = 2
        - page = 2
        - parser = PyMuPDFLoader
        - producer = pdfTeX-1.40.25
        - source = /tmp/gradio/15ac25af8610f21b9ab55252f1944841727ba157/2402.05929.pdf
        - subject = Proceedings of the International Conference on Machine Learning 2024
        - time = 1707724691.929157
        - title = An Interactive Agent Foundation Model
        - total_pages = 22

- **add_chat_history_to_context** (*bool*): 
    Include chat context when performing an action. Not supported when using CLI mode.

- **add_search_to_context** (*bool*): 
    Include web search in context as augmented prompt.

- **context** (*str*): 
    Default context to use (for system pre-context in gradio UI). 
    `context` comes before `chat_conversation` and any document Q/A from `text_context_list`.

- **iinput** (*str*): 
    Default input for instruction-based prompts.

- **allow_upload_to_user_data** (*bool*): 
    Whether to allow file uploads to update shared vector db (UserData or custom user dbs).
    Ensure to pass `user_path` for the files uploaded to be moved to this location for linking.

- **reload_langchain_state** (*bool*): 
    Whether to reload `langchain_modes.pkl` file that contains any new user collections.

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

- **headsize** (*int*):
    Maximum number of characters for head of document document for UI to show.

- **n_jobs** (*int*):
    Number of processors to use when consuming documents (-1 = all, is default).

- **n_gpus** (*int or None*):
    Number of GPUs (None = autodetect).

- **clear_torch_cache_level** (*int*):
    - 0: never clear except where critically required.
    - 1: clear critical.
    - 2: clear aggressively and clear periodically every 20s to free-up GPU memory (may lead to lag in response).

- **use_unstructured** (*bool*):
    Enable unstructured URL loader.

- **use_playwright** (*bool*):
    Enable PlayWright URL loader.

- **use_selenium** (*bool*):
    Enable Selenium URL loader.

- **use_scrapeplaywright** (*bool*):
    Enable Scrape PlayWright URL loader.

- **use_scrapehttp** (*bool*):
    Enable Scrape HTTP URL loader using aiohttp.

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

- **enable_captions** (*bool*):
    Whether to support captions using BLIP for image files as documents.
    Preloads that model if pre_load_image_audio_models=True.

- **enable_llava** (*bool*):
    If LLaVa IP port is set, whether to use response for image ingestion.

- **enable_transcriptions** (*bool*):
    Whether to enable audio transcriptions (YouTube or from files).
    Preloaded if pre_load_image_audio_models=True.

- **pre_load_image_audio_models** (*bool*):
    Whether to preload caption model (True), or load after forking parallel doc loader (False).
    Parallel loading disabled if preload and have images, to prevent deadlocking on CUDA context.
    Recommended if using larger caption model or doing production serving with many users to avoid GPU OOM if many would use model at the same time.
    Also applies to DocTR and ASR models.

- **captions_model** (*str*):
    Which model to use for captions.
    - "Salesforce/blip-image-captioning-base": continue capable.
    - "Salesforce/blip2-flan-t5-xl": question/answer capable, 16GB state.
    - "Salesforce/blip2-flan-t5-xxl": question/answer capable, 60GB state.
    Note: opt-based blip2 are not permissive license due to opt and Meta license restrictions.
    Disabled for CPU since BLIP requires CUDA.

- **caption_gpu** (*bool*):
    If support caption, then use GPU if exists.

- **caption_gpu_id** (*str*):
    Which GPU id to use, if 'auto' then select 0.

- **doctr_gpu** (*bool*):
    If support doctr, then use GPU if exists.

- **doctr_gpu_id** (*str*):
    Which GPU id to use, if 'auto' then select 0.

- **llava_model** (*str*):
    IP:port for h2oai version of LLaVa gradio server for hosted image chat.
    - E.g. http://192.168.1.46:7861.
    - None means no such LLaVa support.

- **llava_prompt** (*str*):
    Prompt passed to LLaVa for querying the image.

- **image_file** (*str*):
    Initial image for UI (or actual image for CLI) Vision Q/A. Or list of images for some models.

- **image_control** (*str*):
    Initial image for UI Image Control.

- **asr_model** (*str*):
    Name of model for ASR, e.g., openai/whisper-medium or openai/whisper-large-v3 or distil-whisper/distil-large-v3 or microsoft/speecht5_asr.
    - whisper-medium uses about 5GB during processing, while whisper-large-v3 needs about 10GB during processing.

- **asr_gpu** (*bool*):
    Whether to use GPU for ASR model.

- **asr_gpu_id** (*str*):
    Which GPU to put ASR model on (only used if preloading model).

- **asr_use_better** (*bool*):
    Whether to use BetterTransformer.

- **asr_use_faster** (*bool*):
    Whether to use faster_whisper package and models (loads normal whisper then unloads it, to get this into pipeline).

- **enable_stt** (*bool*):
    Whether to enable and show Speech-to-Text (STT) with microphone in UI.
    - Note STT model is always preloaded, but if stt_model=asr_model and pre_load_image_audio_models=True, then asr model is used as STT model.

- **stt_model** (*str*):
    Name of model for STT, can be same as asr_model, which will then use the same model for conserving GPU.

- **stt_gpu** (*bool*):
    Whether to use GPU for STT model.

- **stt_gpu_id** (*str*):
    If not using asr_model, then which GPU to go on if using cuda.

- **stt_continue_mode** (*int*):
    How to continue speech with button control.
    - 0: Always append audio regardless of start/stop of recording, so always appends in STT model for full STT conversion.
    - 1: If hit stop, text made so far is saved and audio cleared, so next recording will be separate text conversion.

- **enable_tts** (*bool*):
    Whether to enable TTS.

- **tts_gpu** (*bool*):
    Whether to use GPU if present for TTS.

- **tts_gpu_id** (*str*):
    Which GPU ID to use for TTS.

- **tts_model** (*str*):
    Which model to use for TTS.
    - For microsoft, use 'microsoft/speecht5_tts'.
    - For coqui.ai use one given by doing in python:
      ```python
      from src.tts_coqui import list_models
      list_models()
      ```
      e.g., 'tts_models/multilingual/multi-dataset/xtts_v2'.
    - Note that coqui.ai models are better, but some have non-commercial research license, while microsoft models are MIT.
      So coqui.ai ones can be used for non-commercial activities only, and one should agree to their license, see: https://coqui.ai/cpml
      Commercial use of xtts_v2 should be obtained through their product offering at https://coqui.ai/

- **tts_gan_model** (*str*):
    For microsoft model, which gan model to use, e.g., 'microsoft/speecht5_hifigan'.

- **tts_coquiai_deepspeed** (*bool*):
    For coqui.ai models, whether to use deepspeed for faster inference.

- **tts_coquiai_roles** (*dict*):
    Role dictionary mapping name (key) to wave file (value).
    If None, then just use default from get_role_to_wave_map().

- **chatbot_role** (*str*):
    Default role for coqui models. If 'None', then don't by default speak when launching h2oGPT for coqui model choice.

- **speaker** (*str*):
    Default speaker for microsoft models. If 'None', then don't by default speak when launching h2oGPT for microsoft model choice.

- **tts_language** (*str*):
    Default language for coqui models.

- **tts_speed** (*float*):
    Default speed of TTS, < 1.0 (needs rubberband) for slower than normal, > 1.0 for faster. Tries to keep fixed pitch.

- **tts_action_phrases** (*list*):
    Phrases or words to use as action word to trigger click of Submit hands-free assistant style.
    Set to None or empty list to avoid any special action words.

- **tts_stop_phrases** (*list*):
    Like tts_action_phrases but to stop h2oGPT from speaking and generating.
    NOTE: Action/Stop phrases should be rare but easy (phonetic) words for Whisper to recognize.
          E.g. asking GPT-4 a couple good ones are ['Nimbus'] and ['Yonder'],
          and one can help Whisper by saying "Nimbus Clouds" which still works as "stop word" as trigger.

- **sst_floor** (*float*):
    Floor in wave square amplitude below which ignores the chunk of audio.
    This helps avoid long silence messing up the transcription.

- **jq_schema** (*str*):
    Control json loader. By default '.[]' ingests everything in brute-force way, but better to match your schema.
    See: https://python.langchain.com/docs/modules/data_connection/document_loaders/json#using-jsonloader

- **extract_frames** (*int*):
    How many unique frames to extract from video (if 0, then just do audio if audio type file as well).

- **enable_image** (*bool*):
    Whether to enable image generation model.

- **visible_image_models** (*list*):
    Which image gen models to include.

- **image_gpu_ids** (*list*):
    GPU ids to use for each visible image model.

- **enable_llava_chat** (*bool*):
    Whether to use LLaVa model to chat directly against instead of just for ingestion.

- **max_quality** (*bool*):
    Choose maximum quality ingestion with all available parsers.
    Pro: Catches document when some default parsers would fail.
    Pro: Enables DocTR that has much better OCR than Tesseract.
    Con: Fills DB with results from all parsers, so similarity search gives redundant results.

- **enable_heap_analytics** (*bool*):
    Toggle telemetry.

- **heap_app_id** (*str*):
    App ID for Heap, change to your ID.













