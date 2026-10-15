"""DAPS-CJNet-v2 student with training relations and optional parameter heads."""

from torch import nn

from cjlt.models.label_conditioned import LabelConditionedFeatureBank
from cjlt.models.parameter_heads import LabelGatedParameterHeads


class StudentBModel(nn.Module):
    def __init__(
        self,
        backbone,
        relation_dim=64,
        parameters_per_label=2,
        enable_parameter_heads=False,
        relation_dropout=0.1,
    ):
        super().__init__()
        if not hasattr(backbone, "feature_dim") or not hasattr(backbone, "num_classes"):
            raise ValueError("backbone must expose feature_dim and num_classes")
        self.backbone = backbone
        self.num_classes = int(backbone.num_classes)
        self.feature_dim = int(backbone.feature_dim)
        self.relation_dim = int(relation_dim)
        self.parameters_per_label = int(parameters_per_label)
        self.enable_parameter_heads = bool(enable_parameter_heads)
        self.label_feature_bank = LabelConditionedFeatureBank(
            self.feature_dim,
            self.num_classes,
            relation_dim=self.relation_dim,
            dropout=float(relation_dropout),
        )
        self.parameter_heads = None
        if self.enable_parameter_heads:
            self.parameter_heads = LabelGatedParameterHeads(
                self.num_classes,
                self.relation_dim,
                parameters_per_label=self.parameters_per_label,
            )

    def forward(self, signal, return_aux=False, parameter_stop_gradient=False):
        logits, features = self.backbone(signal, return_features=True)
        if not return_aux:
            return logits
        label_embeddings = self.label_feature_bank(features)
        parameter_prediction = None
        if self.parameter_heads is not None:
            parameter_input = (
                label_embeddings.detach()
                if bool(parameter_stop_gradient)
                else label_embeddings
            )
            parameter_prediction = self.parameter_heads(parameter_input)
        return {
            "logits": logits,
            "features": features,
            "label_embeddings": label_embeddings,
            "parameter_predictions": parameter_prediction,
        }


def count_student_b_deployment_parameters(model):
    if not isinstance(model, StudentBModel):
        raise TypeError("model must be a StudentBModel")
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

