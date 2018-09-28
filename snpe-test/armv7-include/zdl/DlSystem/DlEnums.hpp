//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright 2014-2016 Qualcomm Technologies, Inc. All rights reserved.
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
//
//==============================================================================

#ifndef _DL_ENUMS_HPP_
#define _DL_ENUMS_HPP_


namespace zdl {
namespace DlSystem
{
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * Enumeration of supported target runtimes.
 */
enum class Runtime_t
{
   /// Run the processing on the Snapdragon CPU.
   CPU = 0,

   /// Run the processing on the Adreno GPU.
   GPU,
   /// Run the processing on the Hexagon DSP using fixed point
   DSP
};

/**
 * Enumeration of various performance profiles that can be requested.
 */
enum class PerformanceProfile_t
{
    /// Run in a standard mode
    DEFAULT = 0,

    /// Run in high performance mode
    HIGH_PERFORMANCE,

    /// Run in a power sensitive mode, at the expense of performance.
    POWER_SAVER

};

/**
 * Enumeration of various execution priority hints.
 */
enum class ExecutionPriorityHint_t
{
    /// Normal priority
    NORMAL = 0,

    /// Higher than normal priority
    HIGH,

    /// Lower priority
    LOW

};

/** @} */ /* end_addtogroup c_plus_plus_apis C++*/

/**
 * Enumeration that lists the supported image encoding formats.
 */
enum class ImageEncoding_t
{
   /// For unknown image type. Also used as a default value for ImageEncoding_t.
   UNKNOWN = 0,

   /// The RGB format consists of 3 bytes per pixel: one byte for
   /// Red, one for Green, and one for Blue. The byte ordering is
   /// endian independent and is always in RGB byte order.
   RGB,

   /// The ARGB32 format consists of 4 bytes per pixel: one byte for
   /// Red, one for Green, one for Blue, and one for the alpha channel.
   /// The alpha channel is ignored. The byte ordering depends on the
   /// underlying CPU. For little endian CPUs, the byte order is BGRA.
   /// For big endian CPUs, the byte order is ARGB.
   ARGB32,

   /// The RGBA format consists of 4 bytes per pixel: one byte for
   /// Red, one for Green, one for Blue, and one for the alpha channel.
   /// The alpha channel is ignored. The byte ordering is endian independent
   /// and is always in RGBA byte order.
   RGBA,

   /// The GRAYSCALE format is for 8-bit grayscale.
   GRAYSCALE,

   /// NV21 is the Android version of YUV. The Chrominance is down
   /// sampled and has a subsampling ratio of 4:2:0. Note that this
   /// image format has 3 channels, but the U and V channels
   /// are subsampled. For every four Y pixels there is one U and one V pixel. @newpage
   NV21,

   /// The BGR format consists of 3 bytes per pixel: one byte for
   /// Red, one for Green and one for Blue. The byte ordering is
   /// endian independent and is always BGR byte order.
   BGR
};

}} // namespaces end


#endif
