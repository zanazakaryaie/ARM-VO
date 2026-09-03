#include "SemanticSegmentorFactory.hpp"

#ifdef ARMVO_USE_TENSORRT
#include "SemanticSegmentorTensorRT.hpp"
#else
#include "SemanticSegmentorNcnn.hpp"
#endif

namespace armvo
{

std::unique_ptr<ISemanticSegmentor> SemanticSegmentorFactory::create()
{
#ifdef ARMVO_USE_TENSORRT
    return std::make_unique<SemanticSegmentorTensorRT>();
#else
    return std::make_unique<SemanticSegmentorNcnn>();
#endif
}

} // namespace armvo
