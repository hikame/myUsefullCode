//=============================================================================
//  @@-COPYRIGHT-START-@@
//
//  Copyright 2015-2016 Qualcomm Technologies, Inc. All rights reserved.
//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
//
//  The party receiving this software directly from QTI (the "Recipient")
//  may use this software as reasonably necessary solely for the purposes
//  set forth in the agreement between the Recipient and QTI (the
//  "Agreement"). The software may be used in source code form solely by
//  the Recipient's employees (if any) authorized by the Agreement. Unless
//  expressly authorized in the Agreement, the Recipient may not sublicense,
//  assign, transfer or otherwise provide the source code to any third
//  party. Qualcomm Technologies, Inc. retains all ownership rights in and
//  to the software
//
//  This notice supersedes any other QTI notices contained within the software
//  except copyright notices indicating different years of publication for
//  different portions of the software. This notice does not supersede the
//  application of any third party copyright notice to that third party's
//  code.
//
//  @@-COPYRIGHT-END-@@
//=============================================================================

#ifndef _ITENSOR_FACTORY_HPP
#define _ITENSOR_FACTORY_HPP

#include "ITensor.hpp"
#include "TensorShape.hpp"
#include "ZdlExportDefine.hpp"
#include <istream>

namespace zdl {
    namespace DlSystem
    {
        class ITensor;
        class TensorShape;
    }
}

namespace zdl { namespace DlSystem
{

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * Factory interface class to create ITensor objects.
 */
class ZDL_EXPORT ITensorFactory
{
public:

   /**
    * Creates a new ITensor with uninitialized data. 
    *  
    * The strides for the tensor will match the tensor dimensions 
    * (i.e., the tensor data is contiguous in memory).
    *  
    * @param[in] shape The dimensions for the tensor in which the last
    * element of the vector represents the fastest varying
    * dimension and the zeroth element represents the slowest
    * varying, etc.
    * 
    * @return A pointer to the created tensor or nullptr if creating failed.
    */
   virtual std::unique_ptr<ITensor>
      createTensor(const TensorShape &shape) noexcept = 0;

   /**
    * Creates a new ITensor by loading it from a file.
    *  
    * @param[in] input The input stream from which to read the tensor 
    *                  data.
    *  
    * @return A pointer to the created tensor or nullptr if creating failed.
    *
    */
   virtual std::unique_ptr<ITensor> createTensor(std::istream &input) noexcept = 0;

   /**
    * Create a new ITensor with specific data.
    * (i.e. the tensor data is contiguous in memory). This tensor is
    * primarily used to create a tensor where tensor size can't be
    * computed directly from dimension. One such example is
    * NV21-formatted image, or any YUV formatted image
    *
    * @param[in] shape The dimensions for the tensor in which the last
    * element of the vector represents the fastest varying
    * dimension and the zeroth element represents the slowest
    * varying, etc.
    *
    * @param[in] data The actual data with which the Tensor object is filled.
    *
    * @param[in] dataSize The size of data
    *
    * @return A pointer to the created tensor
    */
   virtual std::unique_ptr<ITensor>
      createTensor(const TensorShape &shape, const unsigned char *data, size_t dataSize) noexcept = 0;
};

}}

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif
