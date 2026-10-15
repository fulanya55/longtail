"""Student A wrapper: original decision head plus label-conditioned auxiliaries."""

from torch import nn

from cjlt.models.label_conditioned import LabelConditionedFeatureBank
from cjlt.models.parameter_heads import LabelGatedParameterHeads


class StudentAModel(nn.Module):
    """Keep deployable logits unchanged while exposing training-only relations."""

    def __init__(
        self,
        backbone,
        relation_dim=64,
        relation_dropout=0.1,
        conditioning_mode="independent",
        enable_parameter_heads=False,
        parameters_per_label=2,
        parameter_hidden_dim=32,
    ):
        super().__init__()
        if not hasattr(backbone, "feature_dim") or not hasattr(backbone, "num_classes"):
            raise ValueError("backbone must expose feature_dim and num_classes")
        self.backbone = backbone
        self.num_classes = int(backbone.num_classes)
        self.feature_dim = int(backbone.feature_dim)
        self.relation_dim = int(relation_dim)
        self.conditioning_mode = str(conditioning_mode)
        self.enable_parameter_heads = bool(enable_parameter_heads)
        self.parameters_per_label = int(parameters_per_label)
        self.label_feature_bank = LabelConditionedFeatureBank(
            feature_dim=self.feature_dim,
            num_labels=self.num_classes,
            relation_dim=self.relation_dim,
            dropout=float(relation_dropout),
            conditioning_mode=self.conditioning_mode,
        )

        self.parameter_heads = None
        if self.enable_parameter_heads:
            self.parameter_heads = LabelGatedParameterHeads(
                num_labels=self.num_classes,
                embedding_dim=self.relation_dim,
                parameters_per_label=self.parameters_per_label,
                hidden_dim=int(parameter_hidden_dim),
            )

    def forward(self, signal, return_aux=False, parameter_stop_gradient=False):
        logits, features = self.backbone(signal, return_features=True)
        if not return_aux:
            return logits
        label_embeddings = self.label_feature_bank(features)
        parameter_predictions = None
        if self.parameter_heads is not None:
            parameter_input = (
                label_embeddings.detach()
                if bool(parameter_stop_gradient)
                else label_embeddings
            )
            parameter_predictions = self.parameter_heads(parameter_input)
        return {
            "logits": logits,
            "features": features,
            "label_embeddings": label_embeddings,
            "parameter_predictions": parameter_predictions,
        }


def count_deployment_parameters(model):
    """Count the inference path after training-only relation branches are stripped."""

    if not isinstance(model, StudentAModel):
        raise TypeError("model must be a StudentAModel")
    modules = [model.backbone]
    if model.enable_parameter_heads:
        modules.extend([model.label_feature_bank, model.parameter_heads])
    return int(
        sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )


def configure_classifier_only_training(model):
    """Freeze the learned representation and keep only the deployment head trainable."""

    if not isinstance(model, StudentAModel):
        raise TypeError("model must be a StudentAModel")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.backbone.classifier.parameters():
        parameter.requires_grad_(True)
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def configure_last_block_classifier_training(model):
    """Adapt only the final temporal block and deployment classifier."""

    if not isinstance(model, StudentAModel):
        raise TypeError("model must be a StudentAModel")
    if len(model.backbone.temporal_backbone) < 1:
        raise ValueError("backbone must expose a final temporal block")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    final_block = model.backbone.temporal_backbone[-1]
    for parameter in final_block.parameters():
        parameter.requires_grad_(True)
    for parameter in model.backbone.classifier.parameters():
        parameter.requires_grad_(True)
    final_block_parameters = int(
        sum(parameter.numel() for parameter in final_block.parameters())
    )
    classifier_parameters = int(
        sum(parameter.numel() for parameter in model.backbone.classifier.parameters())
    )
    return {
        "total": final_block_parameters + classifier_parameters,
        "final_temporal_block": final_block_parameters,
        "classifier": classifier_parameters,
    }


def configure_last_two_blocks_classifier_training(model):
    """Adapt the final two parameterized temporal blocks and classifier."""

    if not isinstance(model, StudentAModel):
        raise TypeError("model must be a StudentAModel")
    temporal_blocks = [
        module
        for module in model.backbone.temporal_backbone
        if any(True for _ in module.parameters())
    ]
    if len(temporal_blocks) < 2:
        raise ValueError("backbone must expose at least two temporal blocks")
    penultimate_block, final_block = temporal_blocks[-2:]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (penultimate_block, final_block, model.backbone.classifier):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    penultimate_parameters = int(
        sum(parameter.numel() for parameter in penultimate_block.parameters())
    )
    final_parameters = int(
        sum(parameter.numel() for parameter in final_block.parameters())
    )
    classifier_parameters = int(
        sum(parameter.numel() for parameter in model.backbone.classifier.parameters())
    )
    return {
        "total": penultimate_parameters + final_parameters + classifier_parameters,
        "penultimate_temporal_block": penultimate_parameters,
        "final_temporal_block": final_parameters,
        "classifier": classifier_parameters,
    }


def keep_frozen_representation_in_eval_mode(model):
    """Prevent frozen BatchNorm/dropout state drift during classifier tuning."""

    if not isinstance(model, StudentAModel):
        raise TypeError("model must be a StudentAModel")
    model.backbone.complex_stem.eval()
    model.backbone.complex_real_bn.eval()
    model.backbone.complex_imag_bn.eval()
    model.backbone.temporal_backbone.eval()
    model.backbone.global_pool.eval()
    for name in ("phase_fusion", "feature_projection"):
        module = getattr(model.backbone, name, None)
        if module is not None:
            module.eval()
    model.label_feature_bank.eval()
    if model.parameter_heads is not None:
        model.parameter_heads.eval()
    model.backbone.classifier.train()


def keep_last_block_adaptation_modes(model):
    """Keep frozen modules fixed while training the final temporal block."""

    if not isinstance(model, StudentAModel):
        raise TypeError("model must be a StudentAModel")
    model.backbone.complex_stem.eval()
    model.backbone.complex_real_bn.eval()
    model.backbone.complex_imag_bn.eval()
    for module in list(model.backbone.temporal_backbone)[:-1]:
        module.eval()
    model.backbone.temporal_backbone[-1].train()
    model.backbone.global_pool.eval()
    for name in ("phase_fusion", "feature_projection"):
        module = getattr(model.backbone, name, None)
        if module is not None:
            module.eval()
    model.label_feature_bank.eval()
    if model.parameter_heads is not None:
        model.parameter_heads.eval()
    model.backbone.classifier.train()


def keep_last_two_block_adaptation_modes(model):
    """Freeze early modules while adapting the final two temporal blocks."""

    if not isinstance(model, StudentAModel):
        raise TypeError("model must be a StudentAModel")
    model.backbone.complex_stem.eval()
    model.backbone.complex_real_bn.eval()
    model.backbone.complex_imag_bn.eval()
    model.backbone.temporal_backbone.eval()
    temporal_blocks = [
        module
        for module in model.backbone.temporal_backbone
        if any(True for _ in module.parameters())
    ]
    if len(temporal_blocks) < 2:
        raise ValueError("backbone must expose at least two temporal blocks")
    for module in temporal_blocks[-2:]:
        module.train()
    model.backbone.global_pool.eval()
    for name in ("phase_fusion", "feature_projection"):
        module = getattr(model.backbone, name, None)
        if module is not None:
            module.eval()
    model.label_feature_bank.eval()
    if model.parameter_heads is not None:
        model.parameter_heads.eval()
    model.backbone.classifier.train()
