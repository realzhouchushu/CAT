import os
from argparse import ArgumentParser
from huggingface_hub import snapshot_download, hf_hub_download

def main(args):
    allow_patterns = []
    repo_id = args.repo_id
    file_name = args.file_name
    repo_type = args.repo_type
    local_dir = args.local_dir
    hf_token = args.hf_token
    allow_patterns = args.allow_patterns

    # -----------------------------------------------------------------------------
    # config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('./configurator.py').read()) # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    if file_name is None and allow_patterns is None:
        # download the whole repo
        model_dir = snapshot_download(repo_id=repo_id,
                                    repo_type=repo_type,
                                    cache_dir=os.path.join(os.path.dirname(__file__), ".cache"),
                                    local_dir=os.path.join(os.path.dirname(__file__), local_dir),
                                    token=hf_token,
                                    )
    elif file_name is not None and allow_patterns is None:
        # download single file
        model_dir = hf_hub_download(repo_id=repo_id, 
                                    repo_type=repo_type,
                                    filename=file_name, 
                                    cache_dir=os.path.join(os.path.dirname(__file__), ".cache"),
                                    local_dir=os.path.join(os.path.dirname(__file__), local_dir),
                                    token=hf_token)
    elif file_name is None and allow_patterns is not None:
        # download files with specifical pattens
        model_dir = snapshot_download(repo_id=repo_id,
                                repo_type=repo_type,
                                allow_patterns=allow_patterns,
                                cache_dir=os.path.join(os.path.dirname(__file__), ".cache"),
                                local_dir=os.path.join(os.path.dirname(__file__), local_dir),
                                token=hf_token,
                                )
    print(f"Download successfully! Save to {model_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--repo_type", type=str, default="model")
    parser.add_argument("--allow_patterns", type=str, default=None, nargs="+")
    parser.add_argument("--local_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()
    main(args)