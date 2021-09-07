import torch
import torch.nn as nn


class Freezer():
    def __init__(self, model):
        self._set_parameter_requires_grad([model], False)
        self.model = model
    
    def freeze_weight(self, phase):
        if phase <= 1:
            self._set_parameter_requires_grad([self.model.context_module, self.model.gating_module], False)
            self._set_parameter_requires_grad([self.model.face_module.classifier_age,
                                                self.model.face_module.classifier_emotion,
                                                self.model.face_module.encoder_att_1,
                                                self.model.face_module.encoder_att_2,
                                                self.model.face_module.encoder_att_3,
                                                self.model.face_module.encoder_att_4,
                                                self.model.face_module.encoder_block_att_1,
                                                self.model.face_module.encoder_block_att_2,
                                                self.model.face_module.encoder_block_att_3], 
                                                True)
            self.model.forward_phase = "face"
        elif phase <= 2:
            self._set_parameter_requires_grad([self.model.context_module, self.model.gating_module], False)
            self._set_parameter_requires_grad([self.model.face_module], True)
            self.model.forward_phase = "face"
        elif phase <= 3:
            self._set_parameter_requires_grad([self.model.face_module, self.model.gating_module], False)
            self._set_parameter_requires_grad([self.model.context_module.classifier_age,
                                                self.model.context_module.classifier_emotion,
                                                self.model.context_module.encoder_att_1,
                                                self.model.context_module.encoder_att_2,
                                                self.model.context_module.encoder_att_3,
                                                self.model.context_module.encoder_att_4,
                                                self.model.context_module.encoder_block_att_1,
                                                self.model.context_module.encoder_block_att_2,
                                                self.model.context_module.encoder_block_att_3],
                                                True)
            self.model.forward_phase = "context"
        elif phase <= 4:
            self._set_parameter_requires_grad([self.model.face_module, self.model.gating_module], False)
            self._set_parameter_requires_grad([self.model.context_module], True)
            self.model.forward_phase = "context"
        elif phase <= 5:
            self.model.forward_phase = "all"
            self._set_parameter_requires_grad([self.model.face_module, self.model.context_module], False)
            self._set_parameter_requires_grad([self.model.gating_module], True)
        else:
            self.model.forward_phase = "all"
            self._set_parameter_requires_grad([self.model], True)

    def _return_model(self):
        return self.model

    @staticmethod
    def _set_parameter_requires_grad(model_list, requires_grad):
        for model in model_list:
            for param in model.parameters():
                param.requires_grad = requires_grad

