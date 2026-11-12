"""Check lazy ML imports without downloading model weights."""


def training_imports():
    from peft import LoraConfig, PeftModel
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    return AutoModel, AutoModelForCausalLM, AutoTokenizer, LoraConfig, PeftModel


def error_message(exc):
    root, seen = exc, set()
    while id(root) not in seen:
        seen.add(id(root))
        parent = root.__cause__ or (None if root.__suppress_context__ else root.__context__)
        if parent is None or id(parent) in seen:
            break
        root = parent
    message = f"{type(exc).__name__}: {exc}"
    if root is not exc:
        message += f"\nRoot cause: {type(root).__name__}: {root}"
    return message


if __name__ == "__main__":
    training_imports()
    print("OK  Transformers model classes and PEFT imports")
