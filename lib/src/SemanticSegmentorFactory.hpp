#pragma once

#include <memory>

#include "ISemanticSegmentor.hpp"

namespace armvo
{

class SemanticSegmentorFactory
{
public:
    static std::unique_ptr<ISemanticSegmentor> create();
};


} // namespace armvo
