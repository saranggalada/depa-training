"""
DP-LoRA integration using Opacus.

Wraps a PEFT (LoRA/QLoRA) model with Opacus ``PrivacyEngine`` so that
only the adapter parameters receive per-sample gradient noise, while
the frozen base model is untouched.

This mirrors the pattern in ``pytrain.dl_train.Train_DL.make_dprivate``
but is adapted for the LLM + PEFT setting.
"""


def maybe_wrap_dp(model, optimizer, train_loader, config):
    """Conditionally wrap model/optimizer/loader with Opacus DP.

    Parameters
    ----------
    model : nn.Module
        The (possibly PEFT-wrapped) model.
    optimizer : torch.optim.Optimizer
        The optimizer.
    train_loader : DataLoader
        The training data loader.
    config : dict
        Either the full validated DEPA config or the pytorch-translated config.
        Looks for ``privacy.enabled``, ``privacy.epsilon``, ``privacy.delta``,
        ``privacy.max_grad_norm``.

    Returns
    -------
    tuple | None
        ``(dp_model, dp_optimizer, dp_loader)`` if DP is enabled,
        otherwise ``None``.
    """
    privacy = config.get("privacy", {})
    if not privacy.get("enabled", False):
        return None

    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
        print("Opacus: model fixed for DP compatibility")

    epsilon = privacy["epsilon"]
    delta = privacy["delta"]
    max_grad_norm = privacy.get("max_grad_norm", 1.0)
    epochs = config.get("training", config).get("epochs", config.get("epochs", 3))

    max_delta = 1.0 / len(train_loader.dataset)
    if delta > max_delta:
        delta = max_delta
        print(f"DP: delta clamped to {delta:.2e} (1/N) to avoid privacy breach")

    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=int(epochs),
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    print(f"DP-LoRA enabled | target_epsilon={epsilon} | delta={delta:.2e} | max_grad_norm={max_grad_norm}")
    return dp_model, dp_optimizer, dp_loader
