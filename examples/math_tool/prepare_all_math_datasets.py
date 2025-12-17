"""
Script to register all math evaluation datasets:
- AIME24 (AIME 2024)
- AIME25 (AIME 2025)
- MATH500 (MATH-500 subset)
- Olympiad (Olympiad Bench)
- AMC23 (AMC 2023)
"""

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_aime2024():
    """AIME 2024 - American Invitational Mathematics Examination 2024"""
    print("Loading AIME 2024...")
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],
            "ground_truth": example["answer"],
            "data_source": "aime2024",
        }
    
    dataset = dataset.map(preprocess_fn, with_indices=True)
    registered = DatasetRegistry.register_dataset("aime2024", dataset, "test")
    print(f"  Registered aime2024 with {len(registered)} problems")
    return registered


def prepare_aime2025():
    """AIME 2025 - American Invitational Mathematics Examination 2025"""
    print("Loading AIME 2025...")
    try:
        # Try loading from HuggingFace
        dataset = load_dataset("yentinglin/aime_2025", split="train")
        
        def preprocess_fn(example, idx):
            return {
                "question": example.get("problem", example.get("question", "")),
                "ground_truth": str(example.get("answer", example.get("solution", ""))),
                "data_source": "aime2025",
            }
        
        dataset = dataset.map(preprocess_fn, with_indices=True)
    except Exception as e:
        print(f"  Warning: Could not load aime_2025 from HuggingFace: {e}")
        print("  Trying alternative source...")
        try:
            dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
            
            def preprocess_fn(example, idx):
                return {
                    "question": example.get("problem", ""),
                    "ground_truth": str(example.get("answer", "")),
                    "data_source": "aime2025",
                }
            
            dataset = dataset.map(preprocess_fn, with_indices=True)
        except Exception as e2:
            print(f"  Error loading AIME 2025: {e2}")
            return None
    
    registered = DatasetRegistry.register_dataset("aime2025", dataset, "test")
    print(f"  Registered aime2025 with {len(registered)} problems")
    return registered


def prepare_math500():
    """MATH-500 - Subset of MATH benchmark"""
    print("Loading MATH-500...")
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        
        def preprocess_fn(example, idx):
            return {
                "question": example["problem"],
                "ground_truth": example["answer"],
                "data_source": "math500",
            }
        
        dataset = dataset.map(preprocess_fn, with_indices=True)
    except Exception as e:
        print(f"  Warning: Could not load MATH-500: {e}")
        print("  Trying alternative source...")
        try:
            # Try loading full MATH and taking subset
            dataset = load_dataset("lighteval/MATH", "all", split="test")
            # Take first 500 or sample
            if len(dataset) > 500:
                dataset = dataset.select(range(500))
            
            def preprocess_fn(example, idx):
                return {
                    "question": example["problem"],
                    "ground_truth": example["solution"],
                    "data_source": "math500",
                }
            
            dataset = dataset.map(preprocess_fn, with_indices=True)
        except Exception as e2:
            print(f"  Error loading MATH-500: {e2}")
            return None
    
    registered = DatasetRegistry.register_dataset("math500", dataset, "test")
    print(f"  Registered math500 with {len(registered)} problems")
    return registered


def prepare_olympiad():
    """Olympiad Bench - Mathematical Olympiad problems"""
    print("Loading Olympiad Bench...")
    try:
        dataset = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="test")
        
        def preprocess_fn(example, idx):
            return {
                "question": example.get("question", example.get("problem", "")),
                "ground_truth": str(example.get("final_answer", example.get("answer", ""))),
                "data_source": "olympiad",
            }
        
        dataset = dataset.map(preprocess_fn, with_indices=True)
    except Exception as e:
        print(f"  Warning: Could not load OlympiadBench: {e}")
        print("  Trying alternative configurations...")
        try:
            dataset = load_dataset("Hothan/OlympiadBench", split="test")
            
            def preprocess_fn(example, idx):
                return {
                    "question": example.get("question", example.get("problem", "")),
                    "ground_truth": str(example.get("final_answer", example.get("answer", ""))),
                    "data_source": "olympiad",
                }
            
            dataset = dataset.map(preprocess_fn, with_indices=True)
        except Exception as e2:
            print(f"  Error loading Olympiad: {e2}")
            return None
    
    registered = DatasetRegistry.register_dataset("olympiad", dataset, "test")
    print(f"  Registered olympiad with {len(registered)} problems")
    return registered


def prepare_amc23():
    """AMC 2023 - American Mathematics Competition 2023 (AMC 12A)"""
    print("Loading AMC 2023...")
    try:
        # Primary source: math-ai/amc23 - 40 problems from 2023 AMC 12A
        dataset = load_dataset("math-ai/amc23", split="test")
        
        def preprocess_fn(example, idx):
            return {
                "question": example.get("question", example.get("problem", "")),
                "ground_truth": str(example.get("answer", "")),
                "data_source": "amc23",
            }
        
        dataset = dataset.map(preprocess_fn, with_indices=True)
    except Exception as e:
        print(f"  Warning: Could not load math-ai/amc23: {e}")
        print("  Trying alternative source...")
        try:
            # Alternative: AI-MO validation set (mixed AMC problems)
            dataset = load_dataset("AI-MO/aimo-validation-amc", split="train")
            
            def preprocess_fn(example, idx):
                return {
                    "question": example.get("problem", example.get("question", "")),
                    "ground_truth": str(example.get("answer", "")),
                    "data_source": "amc23",
                }
            
            dataset = dataset.map(preprocess_fn, with_indices=True)
        except Exception as e2:
            print(f"  Error loading AMC 2023: {e2}")
            return None
    
    registered = DatasetRegistry.register_dataset("amc23", dataset, "test")
    print(f"  Registered amc23 with {len(registered)} problems")
    return registered


def prepare_torl_math():
    """TORL Math - Training dataset for tool-augmented math"""
    print("Loading TORL Math...")
    try:
        dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
        
        def preprocess_fn(example, idx):
            return {
                "question": example["problem"],
                "ground_truth": example["answer"],
                "data_source": "torl_math",
            }
        
        dataset = dataset.map(preprocess_fn, with_indices=True)
        
        # Split for test set (use last 500 as test)
        if len(dataset) > 500:
            test_dataset = dataset.select(range(len(dataset) - 500, len(dataset)))
            train_dataset = dataset.select(range(len(dataset) - 500))
        else:
            test_dataset = dataset
            train_dataset = dataset
        
        train_registered = DatasetRegistry.register_dataset("torl_math", train_dataset, "train")
        test_registered = DatasetRegistry.register_dataset("torl_math", test_dataset, "test")
        print(f"  Registered torl_math with {len(train_registered)} train and {len(test_registered)} test problems")
        return train_registered, test_registered
    except Exception as e:
        print(f"  Error loading TORL Math: {e}")
        return None, None


def prepare_all_datasets():
    """Prepare and register all math evaluation datasets"""
    print("="*60)
    print("Preparing all math evaluation datasets")
    print("="*60)
    
    datasets = {}
    
    # AIME 2024
    datasets["aime2024"] = prepare_aime2024()
    
    # AIME 2025
    datasets["aime2025"] = prepare_aime2025()
    
    # MATH-500
    datasets["math500"] = prepare_math500()
    
    # Olympiad Bench
    datasets["olympiad"] = prepare_olympiad()
    
    # AMC 2023
    datasets["amc23"] = prepare_amc23()
    
    # TORL Math (for training)
    train, test = prepare_torl_math()
    datasets["torl_math_train"] = train
    datasets["torl_math_test"] = test
    
    # Summary
    print("\n" + "="*60)
    print("DATASET REGISTRATION SUMMARY")
    print("="*60)
    
    for name, ds in datasets.items():
        if ds is not None:
            print(f"  ✓ {name}: {len(ds)} examples")
        else:
            print(f"  ✗ {name}: Failed to load")
    
    print("="*60)
    
    return datasets


if __name__ == "__main__":
    datasets = prepare_all_datasets()

