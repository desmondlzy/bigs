import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def load_pretrained_and_set_args(args):
	if args.use_pretrained:
		match args.use_pretrained:
			case "dragon-100000":
				args.dataset_root = os.path.join(
					snapshot_download(repo_id="desmondlzy/bigs-data", repo_type="dataset", allow_patterns="dragon/*", local_dir=Path(__file__).parent.parent / "data/bigs"),
					"dragon",
				)
			case _:
				raise ValueError(f"the pretrained model name `{args.use_pretrained}` not found in the model repo (https://huggingface.co/desmondlzy/bigs/tree/main)")

		args.checkpoint_path = hf_hub_download(repo_id="desmondlzy/bigs", filename=f"{args.use_pretrained}.pth")
				
	return args