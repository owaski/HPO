def register():
    try:
        from vllm import ModelRegistry
    except ImportError:
        # vLLM is not available, skip registration
        return
    
    from nemo_rl.models.generation.sqwen2 import SQwen2ForConditionalGeneration
    ModelRegistry.register_model("SQwen2ForConditionalGeneration", SQwen2ForConditionalGeneration)
    from nemo_rl.models.generation.sqwen3 import SQwen3ForConditionalGeneration
    ModelRegistry.register_model("SQwen3ForConditionalGeneration", SQwen3ForConditionalGeneration)