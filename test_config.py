import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def test_config(cfg: DictConfig):
    print("=" * 60)
    print("Configuration Test")
    print("=" * 60)
    
    # Print full config
    print("\nFull Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Test specific fields
    print("\n" + "=" * 60)
    print("Testing Index Config:")
    print("=" * 60)
    print(f"Index type: {cfg.index.type}")
    print(f"Index type (type): {type(cfg.index.type)}")
    
    try:
        print(f"Query proc: {cfg.index.query_proc}")
    except Exception as e:
        print(f"Query proc error: {e}")
    
    try:
        print(f"Compression: {cfg.index.compression}")
    except Exception as e:
        print(f"Compression error: {e}")
    
    try:
        print(f"Optimization: {cfg.index.optimization}")
    except Exception as e:
        print(f"Optimization error: {e}")

if __name__ == "__main__":
    test_config()