from acp.models.discriminators import veltri_acp_classifier
from acp.models.discriminators import acp_classifier_noCONV

from acp.models.decoders import acp_expanded_decoder
from acp.models.encoders import acp_expanded_encoder
from acp.models.master import master

MODEL_GAREDN = {
    'VeltriACPClassifier': veltri_acp_classifier.VeltriACPClassifier,
    'NoConvACPClassifier': acp_classifier_noCONV.NoConvACPClassifier,
    'ACPExpandedDecoder': acp_expanded_decoder.ACPDecoder,
    'ACPExpandedEncoder': acp_expanded_encoder.ACPEncoder,
    'MasterACPTrainer': master.MasterACPTrainer,
}
-