import sys
sys.path.insert(0, ".")  # or wherever your root is

try:
    import tag.world_model.world_latent_parser.latent_parser
    print("latent_parser: ok")
except Exception as e:
    print(f"latent_parser: FAIL — {e}")

try:
    import tag.world_model.world_latent_model.latent_model
    print("latent_model: ok")
except Exception as e:
    print(f"latent_model: FAIL — {e}")